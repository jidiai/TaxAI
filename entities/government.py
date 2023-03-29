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

        self.tau = 0.263
        self.xi = 0.049
        self.tau_a = 0
        self.xi_a = 0
        self.G = 0

        # self.env = None
        self.debt = 0
        self.next_debt = None
        self.action_dim = entity_args['action_shape']

        self.action_space = Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )


    def reset(self, **custom_cfg):
        # todo 这些参数如何初始化？？
        self.tau = 0.263
        self.xi = 0.049
        self.tau_a = 0
        self.xi_a = 0
        self.G = 0

        self.debt = 0
        return np.array([
                       self.tau,
                       self.xi,
                       self.tau_a,
                       self.xi_a,
                       self.G])


    def get_obs(self):
        pass

    def obs_transfer(self, income, asset):
        # [income_mean, income_std, asset_mean, asset_std, K_{t-1}]

        self.income_mean = np.mean(income)
        self.income_std = np.std(income)

        self.asset_mean = np.mean(asset)
        self.asset_std = np.std(asset)

        obs = np.array([self.income_mean, self.income_std, self.asset_mean, self.asset_std])

        return obs


    def get_actions(self):
        #if controllable, overwritten by the agent module
        pass

    def entity_step(self, env, action=None):
        '''
        action = np.array([0.5, 0.2, 0.02, 0, 1])
        '''
        # action = np.array([0.5, 0.2, 0.02, 0, 1])
        # next state
        self.tau, self.xi, self.tau_a, self.xi_a, self.G = action
        self.sum_tax = sum(env.households_tax)
        self.next_debt = (1 + env.RentRate) * self.debt + self.G - self.sum_tax  # B_{t+1}  # debt 可以为负，买卖国债
        self.debt = copy.copy(self.next_debt)

    def tax_function(self, income, asset):
        # x: input
        def tax_f(x, tau, xi):
            xi = np.clip(xi, 0, 0.9)
            return x - (1 - tau)/(1-xi) * np.power(x, 1-xi)

        income_tax = tax_f(income, self.tau, self.xi)
        asset_tax = tax_f(asset, self.tau_a, self.xi_a)
        return income_tax, asset_tax



