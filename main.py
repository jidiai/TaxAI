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
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # set signle thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    args = parse_args()
    path = args.config
    yaml_cfg = OmegaConf.load(f'./cfg/{path}.yaml')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(yaml_cfg.device_num)

    env = economic_society(yaml_cfg.Environment)
    env.reset()
    done = False
    next_agent_name = 'government'
    action_value = None
    step = 0

    # while not done:
        # if env.agent_selection == 'households':
        #     action_value = np.random.random(size=(5, 2))
        # elif env.agent_selection == 'government':
        #     action_value = np.random.random(size=(5,))
    #
        # action = {env.agent_selection: action_value}
        # action = {'Household': np.random.random(size=(5, 2)),
        #           'government': np.random.random(size=(5,))}
        # global_obs, private_obs, sum_r, r,done = env.step(action)  # o^{-i}_{t+1}, r^i_t(s,a), agent_name^{-i}
        # print("step ",step, "reward:", r, "--done: ", done, "--next agent: ", next_agent_name)
        # step += 1
        # print(f'step {step}')


    # # todo trainer 包括 government + N households
    trainer = agent(env, yaml_cfg.Trainer)
    # start to learn
    print("n_households: ", yaml_cfg.Trainer["n_households"])
    trainer.learn()
    # # close the environment
    # env.close()
