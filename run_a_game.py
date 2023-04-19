import numpy as np

from env.env_core import economic_society
from agents.baseline import agent
from utils.seeds import set_seeds
from arguments import get_args
import os
from omegaconf import OmegaConf


yaml_cfg = OmegaConf.load(f'/home/yansong/Desktop/jidiai/AI-TaxingPolicy/cfg/default.yaml')
yaml_cfg.Trainer["n_households"] = 1000
yaml_cfg.Environment.Entities[1]["entity_args"].n = 1000

env = economic_society(yaml_cfg.Environment)
action_space = env.action_spaces
done = False
global_obs, private_obs = env.reset()
step = 0

while not done:
    actions = {}
    for aid, act_space in action_space.items():
        if 'Household' in aid:
            actions[aid] = np.stack([act_space.sample() for _ in range(env.households.n_households)])
        else:
            actions[aid] = act_space.sample()

    next_global_obs, next_private_obs, gov_reward, house_reward, done = env.step(actions)
    step += 1
    env.render()

    print(f'step = {step}')












