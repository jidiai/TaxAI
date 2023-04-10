from entities.base import BaseEntity
from entities.household import Household
from entities.government import Government
import numpy as np
import math
import torch
from gym.spaces import Box
import copy

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

        self.depreciation_rate = env_args['depreciation_rate']
        self.interest_rate = env_args['interest_rate']
        self.hours_max = env_args['hours_max']
        self.episode_years = env_args['episode_years']
        self.year_per_step = env_args['year_per_step']
        self.consumption_tax_rate = env_args['consumption_tax_rate']
        self.episode_length = self.episode_years/self.year_per_step
        self.step_cnt = 0

        global_obs, private_obs = self.reset()

        self.government.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(global_obs.shape[0],), dtype=np.float32
        )
        self.households.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(global_obs.shape[0] + private_obs.shape[1],), dtype=np.float32   # torch.cat([n_global_obs, private_state, n_gov_action], dim=-1)
        )

    def MarketClear(self):
        # aggregate labor
        self.Lt = np.sum(self.households.e * self.ht)
        # Equilibrium
        self.WageRate = (1 - self.alpha) * np.power(self.Kt/self.Lt, self.alpha)

    def generate_gdp(self):
        # C + X + G  = Yt = sum(labor_income) + Rt*Kt
        gdp = (self.Kt ** self.alpha) * (self.Lt ** (1-self.alpha))
        return gdp

    def action_wrapper(self, actions):
        '''
        input: (-1,1)之间
        output: (0.1, 0.9)
        '''
        new_action = (actions+1)/2
        return np.clip(new_action, 0.001, 0.9)

    def workinghours_wrapper(self, ht):
        return self.hours_max * ht


    def step(self, action_dict):
        self.valid_action_dict = self.is_valid(action_dict)

        # update
        self.households.generate_e_ability()
        self.Kt = copy.copy(self.Kt_next)
        self.Bt = copy.copy(self.Bt_next)
        self.households.at = copy.copy(self.households.at_next)

        self.government.tau, self.government.xi, self.government.tau_a, self.government.xi_a, self.Gt_prob, self.Bt2At = self.action_wrapper(self.valid_action_dict[self.government.name])
        # households action
        multi_actions = self.action_wrapper(self.valid_action_dict[self.households.name])
        saving_p = np.array(multi_actions[:, 0])[:,np.newaxis,...]
        self.workingHours = np.array(multi_actions[:, 1])[:,np.newaxis,...]
        self.ht = self.workinghours_wrapper(self.workingHours)

        # market clear
        self.MarketClear()
        self.GDP = self.generate_gdp()
        Gt = 0.189 * self.GDP

        self.income = self.WageRate * self.households.e * self.ht + self.interest_rate * self.households.at
        income_tax, asset_tax = self.tax_function(self.income, self.households.at)

        post_income = self.income - income_tax
        post_asset = self.households.at - asset_tax
        total_wealth = post_income + post_asset

        # compute tax
        aggregate_consumption = (1 - saving_p) * total_wealth
        choose_consumption = 1/(1 + self.consumption_tax_rate) * aggregate_consumption
        c_scale_range = (self.GDP - Gt)/(np.sum(choose_consumption))
        if c_scale_range >= 1:
            self.consumption = choose_consumption
        else:
            self.consumption = c_scale_range * choose_consumption

        consumption_tax = self.consumption * self.consumption_tax_rate
        self.households.at_next = total_wealth - self.consumption * (1+self.consumption_tax_rate)
        self.tax_array = income_tax + asset_tax + consumption_tax

        self.Bt_next = self.Bt2At * np.sum(self.households.at_next)
        self.Kt_next = (1-self.Bt2At) * np.sum(self.households.at_next)
        lump_sum_transfer = self.Bt_next + np.sum(self.tax_array) - (1 + self.interest_rate) * self.Bt - Gt
        self.households.at_next += self.households.at_next/np.sum(self.households.at_next)*lump_sum_transfer

        # next state
        next_global_state, next_private_state = self.get_obs()
        # terminal
        self.wealth_gini = self.gini_coef(self.households.at_next)
        self.income_gini = self.gini_coef(post_income)

        # reward
        self.households_reward = self.utility_function(self.consumption, self.ht)
        # self.government_reward = np.mean(self.households_reward, axis=0)
        # todo 将GDP当作government objective
        self.government_reward = self.GDP / self.wealth_gini
        self.done = bool(self.wealth_gini > 0.9 or math.isnan(self.government_reward))
        if math.isnan(self.government_reward):
            self.ht = self.workinghours_wrapper(np.ones((self.households.n_households, 1)))
            self.consumption = np.zeros((self.households.n_households, 1)) + 0.001
            self.households_reward = self.utility_function(self.consumption, self.ht)
            self.government_reward = np.mean(self.households_reward, axis=0)

        self.step_cnt += 1

        self.done = self.is_terminal()

        return next_global_state, next_private_state, self.government_reward, self.households_reward, self.done


    def is_valid(self, action_dict):
        return action_dict

    def is_terminal(self):
        if self.done:           #household/government termination
            return self.done

        if self.step_cnt >= self.episode_length:
            return True
        else:
            return False

    def reset(self, **custom_cfg):
        self.government.reset()
        self.households.reset()
        self.Kt_next = np.sum(self.households.at_next) * 0.5
        self.Bt_next = np.sum(self.households.at_next) * 0.5
        self.done = False

        self.Kt = copy.copy(self.Kt_next)
        self.workingHours = np.ones((self.households.n_households,1))/3
        self.ht = self.workinghours_wrapper(self.workingHours)
        self.MarketClear()
        self.income = self.households.e * self.ht
        return self.get_obs()

    def get_obs(self):
        '''
        v0:
            global state: income_mean, income_std, asset_mean, asset_std, wage_rate, rent_rate
            private state: e, a
        '''
        income = self.income
        wealth = self.households.at_next
        income_mean = np.mean(income)
        income_std = np.std(income)

        asset_mean = np.mean(wealth)
        asset_std = np.std(wealth)

        global_obs = np.array([income_mean, income_std, asset_mean, asset_std, self.WageRate, self.Kt, self.Lt])
        private_obs = np.concatenate((self.households.e, wealth), -1)

        return global_obs, private_obs

    def utility_function(self, c_t, h_t):
        # life-time CRRA utility
        if 1-self.households.CRRA == 0:
            u_c = np.log(c_t)
        else:
            u_c = c_t ** (1 - self.households.CRRA) / (1 - self.households.CRRA)
        if 1 + self.households.IFE == 0:
            u_h = np.log(h_t)
        else:
            u_h = ((h_t/1000)**(1 + self.households.IFE)/(1 + self.households.IFE))
        current_utility = u_c - u_h
        return current_utility

    def gini_coef(self, wealths):
        '''
        cite: https://github.com/stephenhky/econ_inequality/blob/master/ginicoef.py
        '''
        cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
        sum_wealths = cum_wealths[-1]
        xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths) - 1)
        yarray = cum_wealths / sum_wealths
        B = np.trapz(yarray, x=xarray)
        A = 0.5 - B
        return A / (A + B)

    def tax_function(self, income, asset):
        # x: input
        def tax_f(x, tau, xi):
            return x - (1 - tau)/(1-xi) * np.power(x, 1-xi)

        income_tax = tax_f(income, self.government.tau, self.government.xi)
        asset_tax = tax_f(asset, self.government.tau_a, self.government.xi_a)
        return income_tax, asset_tax


    def close(self):
        # 待修改
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False




