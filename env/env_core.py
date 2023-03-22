from entities.base import BaseEntity
from entities.household import Household
from entities.government import Government
import numpy as np
import math
import torch

class economic_society:
    name = "wealth distribution economic society"
    '''
    decision:
        government
        market clear
        households
    '''
    def __init__(self, cfg):
        super().__init__()

        for entity_arg in cfg['Entities']:
            if entity_arg['entity_name'] == 'household':
                self.households = Household(entity_arg['entity_args'])
            elif entity_arg['entity_name'] == 'government':
                self.government = Government(entity_arg['entity_args'])

        env_args = cfg['env_core']['env_args']
        self.screen = None  # for rendering
        self.alpha = eval(env_args['alpha'])

        # market
        # todo 后面根据市场均衡计算
        self.WageRate = env_args['WageRate']
        self.RentRate = env_args['RentRate']

        self.possible_agents = cfg['env_core']['possible_agents']  #['government', 'households']
        self.episode_years = env_args['episode_years']
        self.year_per_step = env_args['year_per_step']
        self.episode_length = self.episode_years/self.year_per_step
        self.step_cnt = 0
        self.agent_selection_idx = 0

        # compute
        self.households_tax = self.government.tax_function(self.households.income, self.households.asset)

    @property
    def agent_selection(self):
        return self.possible_agents[self.agent_selection_idx]

    def MarketClear(self):
        self.Lt = torch.sum(self.households.e * self.households.workingHours)
        self.Kt = torch.sum(self.households.asset) - self.government.debt

        # Equilibrium
        self.WageRate = (1 - self.alpha) * np.power(self.Kt/self.Lt, self.alpha)
        self.RentRate = self.alpha * np.power(self.Kt/self.Lt, self.alpha - 1)


    def reset(self, **custom_cfg):
        self.government.reset()
        self.households.reset()
        self.WageRate = 1
        self.RentRate = 0.04
        self.done = False
        return self.get_obs()


    def step(self, action_dict):
        self.valid_action_dict = self.is_valid(action_dict)
        # government step
        self.government.entity_step(self, self.valid_action_dict[self.government.name])
        # households step
        households_utility, self.done = self.households.entity_step(self, self.valid_action_dict[self.households.name])
        next_global_state, next_private_state = self.get_obs()
        self.step_cnt += 1
        self.done = self.is_terminal()

        return next_global_state, next_private_state, np.mean(households_utility, axis=0), households_utility, self.done


    def is_valid(self, action_dict):
        return action_dict

    def is_terminal(self):
        if self.done:           #household/government termination
            return self.done

        if self.step_cnt >= self.episode_length:
            return True
        else:
            return False


    def get_obs(self):
        # get government obs
        income = self.households.income
        asset = self.households.asset
        self.income_mean = np.mean(income)
        self.income_std = np.std(income)

        self.asset_mean = np.mean(asset)
        self.asset_std = np.std(asset)

        global_obs = np.array([self.income_mean, self.income_std, self.asset_mean, self.asset_std, self.WageRate, self.RentRate])
        private_obs = self.households.get_obs()

        return global_obs, private_obs


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




