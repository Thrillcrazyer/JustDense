export CUDA_VISIBLE_DEVICES=0

for model_name in JDS_Mamba
do
  for data_name in ETTh1 ETTh2 ETTm1 ETTm2
  do
    for mask_rate in 0.125 0.25 0.375 0.5
    do

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path $data_name.csv \
  --model_id $data_name'_mask_'$mask_rate \
  --mask_rate $mask_rate \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --d_model 128 \
  --expand 2 \
  --d_conv 4 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001

    done
  done
done