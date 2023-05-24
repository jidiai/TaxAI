import numpy as np

from env.env_core import economic_society
from agents.baseline import agent
from agents.rule_based import rule_agent
from agents.independent_RL import independent_agent
from agents.independent_ppo import ppo_agent
from agents.calibration import calibration_agent
from agents.MF_bilevel import mf_agent
from agents.BMFAC import BMFAC_agent
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
    parser.add_argument("--alg", type=str, default='ac', help="ac, rule_based, independent")
    parser.add_argument('--device-num', type=int, default=1, help='the number of cuda service num')
    parser.add_argument('--n_households', type=int, default=100, help='the number of total households')
    # parser.add_argument('--update_freq', type=int, default=1, help='the number of total households')
    # parser.add_argument('--initial_train', type=int, default=2000, help='the number of total households')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # set signle thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    args = parse_args()
    path = args.config
    yaml_cfg = OmegaConf.load(f'./cfg/{path}.yaml')
    # todo if local run code
    # yaml_cfg = OmegaConf.load(f'D:\\code\\AI-TaxingPolicy\\AI-TaxingPolicy\\cfg\\default.yaml')
    yaml_cfg.Trainer["n_households"] = args.n_households
    yaml_cfg.Environment.Entities[1]["entity_args"].n = args.n_households

    set_seeds(yaml_cfg.seed, cuda=yaml_cfg.Trainer["cuda"])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    env = economic_society(yaml_cfg.Environment)

    if args.alg == "ac":
        trainer = agent(env, yaml_cfg.Trainer)
    elif args.alg == "rule_based":
        trainer = rule_agent(env, yaml_cfg.Trainer)
    elif args.alg == "independent":
        trainer = independent_agent(env, yaml_cfg.Trainer)
    elif args.alg == "mf":
        trainer = mf_agent(env, yaml_cfg.Trainer)
    elif args.alg == "ppo":
        trainer = ppo_agent(env, yaml_cfg.Trainer)
    elif args.alg == "bmfac":
        trainer = BMFAC_agent(env, yaml_cfg.Trainer)
    else:
        trainer = calibration_agent(env, yaml_cfg.Trainer)
    # start to learn
    print("n_households: ", yaml_cfg.Trainer["n_households"])
    trainer.learn()
    # trainer.test()
    # # close the environment
    # env.close()
