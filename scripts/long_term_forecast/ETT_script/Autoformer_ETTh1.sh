export CUDA_VISIBLE_DEVICES=1

for model_name in Autoformer JDAutoformer
do
  for pred_len in 96 192 336 720
  do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_$pred_len \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1
  done
done