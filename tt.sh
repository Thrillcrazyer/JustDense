data_name=ETTm2

for model_name in Autoformer JDAutoformer
do
python -u run.py \
  --train_epochs 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1
done

# # for model_name in iTransformer JDiTransformer
# # do
# # python -u run.py \
# #   --task_name long_term_forecast \
# #   --is_training 1 \
# #   --root_path ./dataset/ETT-small/ \
# #   --data_path $data_name.csv \
# #   --model_id $data_name'_96_720' \
# #   --model $model_name \
# #   --data $data_name \
# #   --features M \
# #   --seq_len 96 \
# #   --label_len 48 \
# #   --pred_len 720 \
# #   --e_layers 2 \
# #   --d_layers 1 \
# #   --factor 3 \
# #   --enc_in 7 \
# #   --dec_in 7 \
# #   --c_out 7 \
# #   --des 'Exp' \
# #   --d_model 128 \
# #   --d_ff 128 \
# #   --itr 1
# # done

# for model_name in Mamba JDMamba
# do
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_$pred_len \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len 720 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --expand 2 \
#   --d_ff 16 \
#   --d_conv 4 \
#   --c_out 7 \
#   --d_model 128 \
#   --des 'Exp' \
#   --itr 1 
# done

# # for model_name in PatchTST JDPatchTST
# # do
# # python -u run.py \
# #   --task_name long_term_forecast \
# #   --is_training 1 \
# #   --root_path ./dataset/ETT-small/ \
# #   --data_path ETTm2.csv \
# #   --model_id ETTm2_96_720 \
# #   --model $model_name \
# #   --data ETTm2 \
# #   --features M \
# #   --seq_len 96 \
# #   --label_len 48 \
# #   --pred_len 720 \
# #   --e_layers 3 \
# #   --d_layers 1 \
# #   --factor 3 \
# #   --enc_in 7 \
# #   --dec_in 7 \
# #   --c_out 7 \
# #   --des 'Exp' \
# #   --n_heads 4 \
# #   --batch_size 128 \
# #   --itr 1
# # done


# for model_name in S_Mamba JDS_Mamba
# do
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_720 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --d_state 2 \
#   --learning_rate 0.00005 \
#   --itr 1
# done

# # for model_name in Transformer JDTransformer
# # do
# # python -u run.py \
# #   --task_name long_term_forecast \
# #   --is_training 1 \
# #   --root_path ./dataset/ETT-small/ \
# #   --data_path ETTm2.csv \
# #   --model_id ETTm2_96_720 \
# #   --model $model_name \
# #   --data ETTm2 \
# #   --features M \
# #   --seq_len 96 \
# #   --label_len 48 \
# #   --pred_len 720 \
# #   --e_layers 2 \
# #   --d_layers 1 \
# #   --factor 1 \
# #   --enc_in 7 \
# #   --dec_in 7 \
# #   --c_out 7 \
# #   --des 'Exp' \
# #   --itr 1
# # done
