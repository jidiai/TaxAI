#!/bin/sh$

python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "gini" --seed 1
python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "gdp" --seed 1
python main.py --device-num 0 --n_households 10 --alg "maddpg" --task "social_welfare" --seed 1