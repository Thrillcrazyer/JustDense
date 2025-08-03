# Copyright (c) 2024, Sukjun Hwang, Aakash Lahoti, Ratish Puduppully, Tri Dao, Albert Gu.

import torch
import torch.nn as nn
from einops import repeat

from .matrix_mixers import (
    Dense, 
    Toeplitz,
    Vandermonde,
    Cauchy,
    LowRank,
    Attention,
    Quasiseparable,
)

class MatrixMixer(nn.Module):
    def __init__(
        self,
        matrix_mixer_type,
        is_data_dependent,
        d_model,
        qk_dim,
        max_seq_len=None, # max_seq_len is necessary for data-independent versions.
        expand=1,
        nheads=8,
        bias=False,
        chunk_size=256,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        self.factory_kwargs=factory_kwargs
        super().__init__()
        self.matrix_mixer_type = matrix_mixer_type
        self.is_data_dependent = is_data_dependent
        self.d_model = d_model
        self.qk_dim = qk_dim
        self.max_seq_len = max_seq_len
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.nheads=nheads
        self.headdim = self.d_inner // self.nheads
        assert self.d_inner % self.headdim == 0
        self.d_state = self.nheads * qk_dim
        self.chunk_size = chunk_size
        matrix_mixer, d_in_proj, conv_dim = self.build_matrix_mixer()
        self.matrix_mixer = matrix_mixer

        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)


        self.act = nn.SiLU()

        self.norm =  nn.LayerNorm(normalized_shape=(self.d_inner,))
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
    def build_matrix_mixer(self):
        if self.matrix_mixer_type == "dense":
            assert not self.is_data_dependent, "Data dependent Dense matrix mixer is not supported."
            matrix_mixer = Dense(
                self.d_model,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
                **self.factory_kwargs
            )
            d_in_proj = 2 * self.d_inner
            conv_dim = self.d_inner
        elif self.matrix_mixer_type == "toeplitz":
            matrix_mixer = Toeplitz(
                self.is_data_dependent,
                self.d_model,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
            )
            d_in_proj = self.d_inner + self.is_data_dependent * (2 * self.nheads)
            conv_dim = self.d_inner + self.is_data_dependent * (2 * self.nheads)
        elif self.matrix_mixer_type == "vandermonde":
            matrix_mixer = Vandermonde(
                self.is_data_dependent,
                self.d_model,
                self.qk_dim,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
            )
            d_in_proj = self.d_inner + self.is_data_dependent * (2 * self.d_state)
            conv_dim = self.d_inner + self.is_data_dependent * (2 * self.d_state)
        elif self.matrix_mixer_type == "cauchy":
            matrix_mixer = Cauchy(
                self.is_data_dependent,
                self.d_model,
                self.qk_dim,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
            )
            d_in_proj = self.d_inner + self.is_data_dependent * (2 * self.d_state)
            conv_dim = self.d_inner + self.is_data_dependent * (2 * self.d_state)
        elif self.matrix_mixer_type == "low_rank":
            matrix_mixer = LowRank(
                self.is_data_dependent,
                self.d_model,
                self.qk_dim,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
            )
            d_in_proj = self.d_inner + self.is_data_dependent * (2 * self.d_state)
            conv_dim = self.d_inner + self.is_data_dependent * (2 * self.d_state)
        elif self.matrix_mixer_type == "attention":
            matrix_mixer = Attention(
                self.is_data_dependent,
                self.d_model,
                self.qk_dim,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
            )
            d_in_proj = self.d_inner + self.is_data_dependent * (2 * self.d_state)
            conv_dim = self.d_inner + self.is_data_dependent * (2 * self.d_state)
        elif self.matrix_mixer_type == "quasiseparable":
            matrix_mixer = Quasiseparable(
                self.is_data_dependent,
                self.d_model,
                self.qk_dim,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
                chunk_size=self.chunk_size,
            )
            # Order: [x, B, C, dt]
            d_in_proj = self.d_inner + self.is_data_dependent * (2 * self.d_state + 2 * self.nheads)
            conv_dim = self.d_inner + self.is_data_dependent * (2 * self.d_state)
        else:
            raise NotImplementedError

        return matrix_mixer, d_in_proj, conv_dim

    def forward(self, u,x1,x2,attn_mask=None,tau=None,delta=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        u_proj = self.in_proj(u)  # (B, L, d_in_proj)

        if self.matrix_mixer_type == "dense":
            y = self.matrix_mixer(u)
            #attn_mask= self.matrix_mixer.M 
            attn_mask=None
            
        elif self.matrix_mixer_type == "toeplitz":
            if self.is_data_dependent:
                x_and_conv = u_proj
                x, forward_conv, reverse_conv = torch.split(
                    x_and_conv,
                    [self.d_inner, self.nheads, self.nheads],
                    dim=-1
                )
                y = self.matrix_mixer(x, forward_conv=forward_conv, reverse_conv=reverse_conv)
            else:
                y = self.matrix_mixer(u)

        elif self.matrix_mixer_type in ["vandermonde", "cauchy", "attention"]:
            if self.is_data_dependent:
                vqk = u_proj
                v, q, k = torch.split(vqk, [self.d_inner, self.d_state, self.d_state], dim=-1)
                v = self.act(v)
                y = self.matrix_mixer(v, q=q, k=k)

            else:
                x =u_proj
                y = self.matrix_mixer(u)

        elif self.matrix_mixer_type == "low_rank":
            if self.is_data_dependent:
                vqk = u_proj
                v, q, k = torch.split(vqk, [self.d_inner, self.d_state, self.d_state], dim=-1)
                y = self.matrix_mixer(v, q=q, k=k)
            else:
                x = u_proj
                y = self.matrix_mixer(u)

        elif self.matrix_mixer_type == "quasiseparable":
            if self.is_data_dependent:
                vqk, dt = torch.split(
                    u_proj,
                    [self.d_inner + 2 * self.d_state, 2 * self.nheads],
                    dim=-1
                )
                #dt = dt + repeat(self.matrix_mixer.dt_bias, 'h -> (2 h)')
                dt = repeat(self.matrix_mixer.dt_bias, 'h -> b l (2 h)', b=batch, l=seqlen)
                v, qk = torch.split(vqk, [self.d_inner, 2 * self.d_state], dim=-1)
                y = self.matrix_mixer(v, qk, dt)
                
            else:
                v = u_proj
                dt = repeat(self.matrix_mixer.dt_bias, 'h -> b l (2 h)', b=batch, l=seqlen)
                qk = repeat(self.matrix_mixer.BC, 'l d -> b l d', b=batch)
                y = self.matrix_mixer(v, qk, dt)        
        out=y
        
        if self.matrix_mixer_type!='dense':
            attn_mask=None
        
        return out,attn_mask