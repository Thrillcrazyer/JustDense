export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

for model_name in JDPatchTST PatchTST
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id $model_name'_336_96' \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 128 \
  --itr 1 \
  --n_heads 1
done