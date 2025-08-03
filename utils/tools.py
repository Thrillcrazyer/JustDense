import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visualize_attn_map(attn, name='./attn_map.jpg', np_name='./attn_map.npy', dpi=600):
    print("FINAL ATTN MAP SHAPE: ", attn.shape)
    #attn = attn.squeeze()             # [32, 11, 11]
    #attn = attn[0,:,:]           # [11, 11]
    if len(attn.shape)==3:
        attn = attn.mean(dim=0)   
    if len(attn.shape)==4:
        attn = attn.mean(dim=(0, 1))
        # [11, 11] 평균을 통해 단일 어텐션 맵 생성
    plt.figure(figsize=(5, 5), dpi=dpi)
    plt.imshow(attn.detach().cpu().numpy(), cmap='viridis',vmin=0, vmax=0.4)  # 어텐션 맵 시각화
    plt.axis('off')                    # 축 제거 (선택)
    plt.tight_layout(pad=0)            # 여백 최소화
    plt.savefig(name, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    np.save(np_name, attn.detach().cpu().numpy())


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def test_params_flop(model, x_shape, label_len=48, pred_len=720):
    """
    Handle models with multiple forward arguments and correct input shapes
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            batch_size, seq_len, features = x.shape
            dec_len = label_len + pred_len
            x_mark_enc = torch.zeros(batch_size, seq_len, 4).to(x.device)
            x_dec = torch.zeros(batch_size, dec_len, features).to(x.device)
            x_mark_dec = torch.zeros(batch_size, dec_len, 4).to(x.device)
            return self.model(x, x_mark_enc, x_dec, x_mark_dec)

    try:
        from ptflops import get_model_complexity_info
        wrapped_model = ModelWrapper(model)
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(
                wrapped_model.cuda(),
                x_shape[1:],  # Remove batch dimension
                as_strings=True,
                print_per_layer_stat=False
            )
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    except Exception as e:
        print(f"FLOP calculation failed: {e}")
        print("Only parameter count is available.")
    return macs,model_params

