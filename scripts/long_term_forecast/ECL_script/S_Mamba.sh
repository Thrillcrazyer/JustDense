export CUDA_VISIBLE_DEVICES=0

for model_name in JDS_Mamba
do
  for pred_len in 96 192 336 720
  do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_96_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 2 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 512 \
      --d_state 16 \
      --train_epochs 5 \
      --batch_size 16 \
      --learning_rate 0.0005 \
      --itr 1
  done
done