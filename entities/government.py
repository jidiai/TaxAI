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
        self.tau = self.entity_args["tau"]
        self.xi = self.entity_args["xi"]
        self.tau_a = self.entity_args["tau_a"]
        self.xi_a = self.entity_args["xi_a"]


    def obs_transfer(self, income, asset):

        self.income_mean = np.mean(income)
        self.income_std = np.std(income)

        self.asset_mean = np.mean(asset)
        self.asset_std = np.std(asset)

        obs = np.array([self.income_mean, self.income_std, self.asset_mean, self.asset_std])

        return obs



