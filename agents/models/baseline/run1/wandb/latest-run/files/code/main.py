import numpy as np

from env.env_core import economic_society
from agents.baseline import agent
from utils.seeds import set_seeds
from arguments import get_args
import os
import torch
import yaml
import argparse
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='default')
    parser.add_argument('--device-num', type=int, default=1, help='the number of cuda service num')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # set signle thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    args = parse_args()
    path = args.config
    yaml_cfg = OmegaConf.load(f'./cfg/{path}.yaml')
    set_seeds(yaml_cfg.seed, cuda=yaml_cfg.Trainer["cuda"])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    env = economic_society(yaml_cfg.Environment)

    trainer = agent(env, yaml_cfg.Trainer)
    # start to learn
    print("n_households: ", yaml_cfg.Trainer["n_households"])
    trainer.learn()
    # trainer.test()
    # # close the environment
    # env.close()
