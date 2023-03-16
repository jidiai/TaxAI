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

        # government information
        self.tau = 0.5
        self.xi = 0.2
        self.tau_a = 0.02
        self.xi_a = 0
        self.G = 0
        self.Kt = 0

        self.possible_agents = ['households', 'government']
        self.episode_years = 100
        self.year_per_step = 1
        self.episode_length = self.episode_years/self.year_per_step
        self.step_cnt = 0
        self.agent_selection_idx = 0

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
        self.households.reset()
        self.government.reset()
        self.done = False

    def step(self, action_dict):
        self.valid_action_dict = self.is_valid(action_dict)
        obs, reward = self.run()

        self.step_cnt += 1

        self.done = self.is_terminal()
        # reward = 0
        # print(f'step {self.step_cnt}')

        return obs, reward, self.done, self.agent_selection, ""




    def run(self):
        """
        run a year
        """
        ################# entity step  ############################
        # for agent_name in self.possible_agents:                 #this take step in turns
        for agent_name, agent_action in self.valid_action_dict.items():
            _state, _reward, _done = getattr(getattr(self, agent_name), 'entity_step')(self, agent_action)  # entity step
            if _done:
                self.done = _done
                break               #break the for loop if done??
        ################## after each entity have taken actions, proceed an environment step ##########################3

        #TODO: global state change


        ##################### get obs ################################3
        self.agent_selection_idx = (self.agent_selection_idx+1)%len(self.possible_agents)  # take turn

        rets = {}
        next_agent_name = self.agent_selection

        rets[next_agent_name] = getattr(getattr(self, next_agent_name), 'get_obs')(self)

        return rets, _reward


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




