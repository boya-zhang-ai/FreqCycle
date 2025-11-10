if [ ! -d "./logs1" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs1/FreqCycle" ]; then
    mkdir ./logs1/FreqCycle
fi
if [ ! -d "./logs1/FreqCycle/Electricity" ]; then
    mkdir ./logs1/FreqCycle/Electricity
fi
model_name=FreqCycle

root_path_name=./data
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom
seg_window=6
seg_stride=6

model_type='mlp'
seq_len=96

for window_type in 'rect' 
do
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
      --enc_in 321 \
      --cycle 168 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 15 \
      --itr 1 \
      --batch_size 64 \
      --learning_rate 0.005 \
      --seg_window $seg_window \
      --seg_stride $seg_stride \
      --window_type $window_type \
      --random_seed $random_seed >logs1/FreqCycle/Electricity/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$random_seed'_'$seg_window'_'$seg_stride'_'$window_type.log 

done
done
done

