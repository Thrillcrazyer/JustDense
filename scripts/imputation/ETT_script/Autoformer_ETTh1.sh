export CUDA_VISIBLE_DEVICES=0

for model_name in Autoformer JDAutoformer
do
  for mask_rate in 0.125 0.25 0.375 0.5
  do
    python -u run.py \
      --task_name imputation \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_mask_$mask_rate \
      --mask_rate $mask_rate \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --label_len 0 \
      --pred_len 0 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 16 \
      --d_model 128 \
      --d_ff 128 \
      --des 'Exp' \
      --itr 1 \
      --top_k 5 \
      --learning_rate 0.001
  done
done
