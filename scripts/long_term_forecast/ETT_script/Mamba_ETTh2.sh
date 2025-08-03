export CUDA_VISIBLE_DEVICES=1

for model_name in Mamba JDMamba
do
for pred_len in 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $pred_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 1 \
  --enc_in 7 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 7 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \

done
done