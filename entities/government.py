from entities.base import BaseEntity
from utils.episode import EpisodeKey
import math
import copy

class Government(BaseEntity):
    name='government'

    def __init__(self, entity_args):
        super().__init__()
        # todo entity_args 是设定好的嘛？？ 这些是 action
        # self.tau = entity_args.income_tax
        # self.xi = entity_args.income_tax_slope
        # self.tau_a = entity_args.wealth_tax
        # self.xi_a = entity_args.income_tax_slope
        # self.G = entity_args.government_spending


    def reset(self, **custom_cfg):
        # self.tau =
        pass


    def get_obs(self):
        pass


    def get_actions(self):
        #if controllable, overwritten by the agent module
        pass

    def entity_step(self, action):
        #abilitiy transition


        # next state
        self.tau, self.xi, self.tau_a, self.xi_a, self.G = action
        pass

    def tax_function(self, tau, xi, x):
        # x: input
        return x - (1 - tau)/(1-xi) * math.pow(x, 1-xi)


