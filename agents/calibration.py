import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import os,sys
import wandb
import pandas as pd
sys.path.append(os.path.abspath('../..'))

from agents.models import mlp
from agents.log_path import make_logpath
from utils.experience_replay import replay_buffer

from scipy.optimize import minimize

torch.autograd.set_detect_anomaly(True)

def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class calibration_agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args

        self.eval_env = copy.copy(envs)
        # get the action max
        self.gov_action_max = self.envs.government.action_space.high[0]
        self.hou_action_max = self.envs.households.action_space.high[0]
        #
        self.model_path, _ = make_logpath(algo="rule_based")
        save_args(path=self.model_path, args=self.args)

        self.h_mean = (self.envs.households.real_income / (self.envs.WageRate * self.envs.households.e)).mean()
        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)

    def calculate_h(self, W, e, I):
        def objective(h):
            L = (e * h).sum()
            return np.abs(W - (1 - self.envs.alpha) * (self.envs.Kt / L) ** self.envs.alpha)


        # bounds = [(0, 1) for _ in range(len(e))]
        h0 = I / (W * e)
        res = minimize(objective, h0)
        return res.x * np.sum(e * I) / np.sum(e * res.x)

    def learn(self):

        gov_action = np.array([0.263, 0.049, 0.02, 0, 0, 0.4])
        temp = np.ones((self.args.n_households, 2))
        temp[:, 0] = 0.9

        h = self.h_mean * temp[:, 1]
        for update_i in range(100):
            temp[:, 1] = h
            hou_action = temp

            action = {self.envs.government.name: (gov_action),
                      self.envs.households.name: hou_action}


            next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)

            W = self.envs.WageRate
            e = self.envs.households.e
            I = self.envs.households.real_income

            # 计算 h
            old_h = h
            h = self.calculate_h(W, e, I)
            # print(h)  # 输出结果
            print("h max:", h.max(), "h min:", h.min())
            # self.envs.households.lorenz_curve(h)

            error = np.abs(old_h - h).mean()
            if error < 1:
                break


