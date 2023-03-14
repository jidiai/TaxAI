from entities.base import BaseEntity
from entities.household import Household
from entities.government import Government
import numpy as np
import math
import torch

class economic_society(BaseEntity):
    name = "wealth distribution economic society"
    '''
    decision:
        government
        market clear
        households
    '''
    def __init__(self, entity_args):
        super().__init__()
        self.households = Household(entity_args)
        self.government = Government(entity_args)
        self.screen = None  # for rendering
        self.alpha = entity_args.alpha

        # market
        # todo 后面根据市场均衡计算
        self.WageRate = 1
        self.RentRate = 0.04
        # self.next_WageRate = None
        # self.next_RentRate = 0.04

        # government information
        self.tau = 0.5
        self.xi = 0.2
        self.tau_a = 0.02
        self.xi_a = 0
        self.G = 0



    def MarketClear(self):
        self.Lt = torch.sum(self.households.e * self.households.workingHours)
        self.Kt = torch.sum(self.households.asset) - self.government.debt

        # Equilibrium
        self.WageRate = (1 - self.alpha) * np.power(self.Kt/self.Lt, self.alpha)
        self.RentRate = self.alpha * np.power(self.Kt/self.Lt, self.alpha - 1)


    def reset(self, **custom_cfg):
        self.households.reset()
        self.government.reset()


    def get_obs(self):
        pass

    def get_actions(self):
        # if controllable, overwritten by the agent module
        pass

    def entity_step(self, action):
        # abilitiy transition



        pass

    def close(self):
        # 待修改
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False




