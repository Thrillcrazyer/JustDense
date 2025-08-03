export CUDA_VISIBLE_DEVICES=0

for model_name in PatchTST JDPatchTST
do
  for pred_len in 96 192 336 720
  do
    if [[ "$model_name" == "PatchTST" && "$pred_len" != "720" ]]
then
  continue
fi

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
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --batch_size 16 \
      --itr 1
  done
done