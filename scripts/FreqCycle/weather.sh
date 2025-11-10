if [ ! -d "./logs1" ]; then
    mkdir ./logs1
fi

if [ ! -d "./logs1/FreqCycle" ]; then
    mkdir ./logs1/FreqCycle
fi
if [ ! -d "./logs1/FreqCycle/weather" ]; then
    mkdir ./logs1/FreqCycle/weather
fi
model_name=FreqCycle

root_path_name=./data
data_path_name=weather.csv
model_id_name=Weather
data_name=custom
seg_window=8
seg_stride=8

model_type='mlp'
seq_len=96

for pred_len in 96 192 336 720
do
for random_seed in 2026 
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
      --enc_in 21 \
      --cycle 144 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 15 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.01 \
      --seg_window $seg_window \
      --seg_stride $seg_stride \
      --random_seed $random_seed >logs1/FreqCycle/weather/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$random_seed'_'$seg_window'_'$seg_stride.log 

done
done

