if [ ! -d "./logs1" ]; then
    mkdir ./logs1
fi

if [ ! -d "./logs1/FreqCycle" ]; then
    mkdir ./logs1/FreqCycle
fi
if [ ! -d "./logs1/FreqCycle/ETTh2" ]; then
    mkdir ./logs1/FreqCycle/ETTh2
fi
model_name=FreqCycle

root_path_name=./data/ETT
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
model_type='mlp'
seq_len=96
seg_window=6
seg_stride=6
for window_type in 'rect'
do
for pred_len in 96 192 
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
      --batch_size 256 \
      --d_model 128 \
      --learning_rate 0.005 \
      --seg_window $seg_window \
      --seg_stride $seg_stride \
      --window_type $window_type \
      --random_seed $random_seed >logs1/FreqCycle/ETTh2/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$random_seed'_'$seg_window'_'$seg_stride'_'$window_type.log 

done
done
done 

for window_type in 'rect'
do
for pred_len in  336 720
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
      --batch_size 128\
      --d_model 512 \
      --learning_rate 0.001 \
      --seg_window $seg_window \
      --seg_stride $seg_stride \
      --window_type $window_type \
      --random_seed $random_seed >logs1/FreqCycle/ETTh2/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$random_seed'_'$seg_window'_'$seg_stride'_'$window_type.log 

done
done
done 

