#!/bin/sh$

#python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "gini" --seed 2 --hidden_size 64 --q_lr 3e-3 --p_lr 3e-3 --batch_size 32
#python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "gini" --seed 2 --hidden_size 128 --q_lr 3e-3 --p_lr 3e-3 --batch_size 32
#python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "gini" --seed 2 --hidden_size 256 --q_lr 3e-3 --p_lr 3e-3 --batch_size 32
#
#python main.py --device-num 1 --n_households 100 --alg "ppo" --task "gdp" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --device-num 0 --n_households 10000 --alg "ppo" --task "gdp" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --device-num 0 --n_households 100 --alg "ppo" --task "social_welfare" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128

#python main.py --device-num 0 --n_households 100 --alg "bmfac" --task "gdp" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
#python main.py --device-num 0 --n_households 100 --alg "bmfac" --task "gini" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
#python main.py --device-num 0 --n_households 100 --alg "bmfac" --task "social_welfare" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128

#
#dist="bmfac"
#n=10
#device=1
#seed=1
#obj_task=""
#lr=3e-4
#batch_size=128
#
#for update_freq in {10,20,30};
#do
#  for update_cycles in {100,10,1000};
#  do
#    for initial_train in {10,100,200};
#    do
#      let real_update_cycles=update_cycles
#      let real_update_freq=update_freq
#      let real_initial_train=initial_train
#      echo "real_seed is ${seed}"
#      echo "device_num is ${device}"
#      echo "alg is ${dist}"
#      echo "task is ${obj_task}"
#      echo "n is ${n}"
#      echo "update_cycles is ${real_update_cycles}"
#      echo "update_freq is ${real_update_freq}"
#      echo "initial_train is ${real_initial_train}"
#      echo "q_lr is ${lr}"
#      echo "batch_size is ${batch_size}"
#      CUDA_VISIBLE_DEVICES=-1 python main.py --device-num $device --n_households $n --alg $dist --task $obj_task --seed $seed --update_cycles $real_update_cycles --update_freq $real_update_freq --initial_train $real_initial_train
#    done
#  done
#done
#
