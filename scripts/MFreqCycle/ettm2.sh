if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/MFreqCycle" ]; then
    mkdir ./logs/MFreqCycle
fi
if [ ! -d "./logs/MFreqCycle/ETTm2" ]; then
    mkdir ./logs/MFreqCycle/ETTm2
fi
model_name=MFreqCycle

root_path_name=./data/ETT
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=customM


model_type='mlp'
seq_len=672

for pred_len in 96 192 336 720
do
for random_seed in 2024 
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 15 \
      --itr 1 \
      --batch_size 256 \
      --learning_rate 0.005 \
      --seg_num 3 \
      --Cycle_num 2 \
      --every_Cycle 672 96 \
      --recent_num 672 96 \
      --Cycle_stride 4 1 \
      --Cycle_window 8 1 \
      --target OT \
      --random_seed $random_seed >logs/MFreqCycle/ETTm2/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

done
done


