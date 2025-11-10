if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/MFreqCycle" ]; then
    mkdir ./logs/MFreqCycle
fi
if [ ! -d "./logs/MFreqCycle/ETTh2" ]; then
    mkdir ./logs/MFreqCycle/ETTh2
fi
model_name=MFreqCycle

root_path_name=./data/ETT
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=customM


model_type='mlp'
seq_len=168
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
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 15 \
      --itr 1 \
      --batch_size 64 \
      --learning_rate 0.005 \
      --seg_num 3 \
      --Cycle_num 2 \
      --every_Cycle 168 24 \
      --recent_num 168 96 \
      --Cycle_stride 2 1 \
      --Cycle_window 4 1 \
      --random_seed $random_seed >logs/MFreqCycle/ETTh2/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 

done
done


