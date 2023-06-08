#!/bin/sh$

#python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "gini" --seed 2 --hidden_size 64 --q_lr 3e-3 --p_lr 3e-3 --batch_size 32
#python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "gini" --seed 2 --hidden_size 128 --q_lr 3e-3 --p_lr 3e-3 --batch_size 32
#python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "gini" --seed 2 --hidden_size 256 --q_lr 3e-3 --p_lr 3e-3 --batch_size 32
#
#python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "gini" --seed 2 --hidden_size 64 --q_lr 3e-3 --p_lr 3e-3 --batch_size 32
#python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "gini" --seed 2 --hidden_size 64 --q_lr 3e-3 --p_lr 3e-3 --batch_size 32
#python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "gdp" --seed 2
#python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "social_welfare" --seed 2



dist="maddpg"
n=10
device=1
seed=1
obj_task="gdp"
lr=3e-3
batch_size=128
real_hidden_size=128

for update_cycles in {100,10,1000};
do
  let real_update_cycles=update_cycles
  echo "real_seed is ${seed}"
  echo "device_num is ${device}"
  echo "alg is ${dist}"
  echo "task is ${obj_task}"
  echo "n is ${n}"
  echo "hidden_size is ${real_hidden_size}"
  echo "update_cycles is ${real_update_cycles}"
  echo "q_lr is ${lr}"
  echo "batch_size is ${batch_size}"
  CUDA_VISIBLE_DEVICES=-1 python main.py --device-num $device --n_households $n --alg $dist --task $obj_task --seed $seed --hidden_size $real_hidden_size --q_lr $lr --p_lr $lr --batch_size $batch_size --update_cycles $real_update_cycles
done

