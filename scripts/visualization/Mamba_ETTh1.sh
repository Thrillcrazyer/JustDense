export CUDA_VISIBLE_DEVICES=1

for pred_len in 96
do
for model_name in Mamba JDMamba
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id $model_name'_96_96' \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $pred_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 1 \
  --d_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --expand 1 \
  --d_ff 16 \
  --d_conv 1 \
  --d_model 4 \
  --des 'Exp' \
  --itr 1 

done
done