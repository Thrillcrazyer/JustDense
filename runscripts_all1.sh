# ./run_all.sh --path /workspace/scripts/anomaly_detection/MSL
# ./run_all.sh --path /workspace/scripts/anomaly_detection/SMAP
# ./run_all.sh --path /workspace/scripts/anomaly_detection/SWAT

# ./run_all.sh --path /workspace/scripts/imputation/ETT_script
# ./run_all.sh --path /workspace/scripts/imputation/Weather_script

#./run_all.sh --path /workspace/scripts/long_term_forecast/ECL_script

bash /workspace/scripts/long_term_forecast/ECL_script/PatchTST.sh
bash /workspace/scripts/long_term_forecast/ECL_script/S_Mamba.sh
bash /workspace/scripts/long_term_forecast/ECL_script/Transformer.sh

./run_all.sh --path /workspace/scripts/long_term_forecast/ETT_script

# bash /workspace/scripts/long_term_forecast/Traffic_script/iTransformer.sh
# bash /workspace/scripts/long_term_forecast/ETT_script/iTransformer_ETTh2.sh
# bash /workspace/scripts/long_term_forecast/ECL_script/iTransformer.sh
# bash /workspace/scripts/long_term_forecast/Weather_script/iTransformer.sh