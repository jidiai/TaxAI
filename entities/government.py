from entities.base import BaseEntity
from utils.episode import EpisodeKey
import math
import copy
import numpy as np
from gym.spaces import Box

class Government(BaseEntity):
    name='government'

    def __init__(self, entity_args):
        super().__init__()
        self.entity_args = entity_args

        self.reset()
        self.action_dim = entity_args['action_shape']

        self.action_space = Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )


    def reset(self, **custom_cfg):
        # todo 这些参数如何初始化？？
        self.tau = self.entity_args["tau"]
        self.xi = self.entity_args["xi"]
        self.tau_a = self.entity_args["tau_a"]
        self.xi_a = self.entity_args["xi_a"]
        # self.G = self.entity_args["G"]


    # def get_obs(self):
    #     pass

    def obs_transfer(self, income, asset):
        # [income_mean, income_std, asset_mean, asset_std, K_{t-1}]

        self.income_mean = np.mean(income)
        self.income_std = np.std(income)

        self.asset_mean = np.mean(asset)
        self.asset_std = np.std(asset)

        obs = np.array([self.income_mean, self.income_std, self.asset_mean, self.asset_std])

        return obs

    #
    # def get_actions(self):
    #     #if controllable, overwritten by the agent module
    #     pass

    # def entity_step(self, env, action=None):
    #     '''
    #     action = np.array([0.5, 0.2, 0.02, 0, 1])
    #     '''
    #     # action = np.array([0.5, 0.2, 0.02, 0, 1])
    #     # next state
    #     self.debt = copy.copy(self.next_debt)
    #     self.tau, self.xi, self.tau_a, self.xi_a, self.G = action
    #     self.sum_tax = np.sum(env.households_tax)
    #     self.next_debt = (1 + env.InterestRate) * self.debt + self.G*self.G_scale - self.sum_tax  # B_{t+1}  # debt 可以为负，代表国家有净财富；但是大多都是负的
    #     # todo assume Bt+1 = Bt
    #     # self.G = self.sum_tax




