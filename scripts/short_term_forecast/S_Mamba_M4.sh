export CUDA_VISIBLE_DEVICES=0

for model_name in S_Mamba JDS_Mamba
do
# d state 2
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.001 \
  --loss 'SMAPE'  \
  --itr 1

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.001 \
  --loss 'SMAPE'  \
  --itr 1

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.001 \
  --loss 'SMAPE'  \
  --itr 1

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.001 \
  --loss 'SMAPE'  \
  --itr 1

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.001 \
  --loss 'SMAPE'  \
  --itr 1

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.001 \
  --loss 'SMAPE'  \
  --itr 1

done