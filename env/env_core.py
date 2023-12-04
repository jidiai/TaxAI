
from entities.household import Household
from entities.government import Government
import numpy as np
import math
import torch
from gym.spaces import Box
import copy
import pygame
import sys
import os


from pathlib import Path
ROOT_PATH = str(Path(__file__).resolve().parent.parent)

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
        self.consumption_tax_rate = env_args['consumption_tax_rate']
        self.gini_weight = env_args['gini_weight']
        self.gov_task = env_args['gov_task']
        # self.episode_length = self.episode_years/self.year_per_step
        self.episode_length = 300
        self.step_cnt = 0
        self.xi_max = 0.05
        self.tau_max = 0.2

        global_obs, private_obs = self.reset()

        self.government.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(global_obs.shape[0],), dtype=np.float32
        )
        self.households.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(global_obs.shape[0] + private_obs.shape[1],), dtype=np.float32   # torch.cat([n_global_obs, private_state, n_gov_action], dim=-1)
        )

        self.display_mode = False

    @property
    def action_spaces(self):
        return {self.households.name: self.households.action_space,
                self.government.name: self.government.action_space}

    @property
    def observation_spaces(self):
        return {self.households.name: self.households.observation_space,
                self.government.name: self.government.observation_space}

    def MarketClear(self):
        # aggregate labor
        self.Lt = np.sum(self.households.e * self.ht)
        # Equilibrium
        self.WageRate = (1 - self.alpha) * np.power(self.Kt/self.Lt, self.alpha)
        # capital-labor ratio =  (self.alpha / self.interest_rate)**(1/(1-self.alpha))

    def generate_gdp(self):
        # C + X + G  = Yt = sum(labor_income) + Rt*Kt
        gdp = (self.Kt ** self.alpha) * (self.Lt ** (1-self.alpha))
        self.per_household_gdp = gdp / self.households.n_households
        return gdp

    def action_wrapper(self, actions):
        '''
        input: (-1,1)之间
        output: (0, 0.9999)
        '''
        return np.clip((actions+1)/2, 0.001, 0.99)

    def workinghours_wrapper(self, ht):
        return self.hours_max * ht

    def step(self, action_dict):
        self.old_per_gdp = copy.copy(self.per_household_gdp)
        self.valid_action_dict = self.is_valid(action_dict)

        # update
        self.households.generate_e_ability()
        self.Kt = copy.copy(self.Kt_next)
        self.Bt = copy.copy(self.Bt_next)
        self.households.at = copy.copy(self.households.at_next)

        self.government.tau, self.government.xi, self.government.tau_a, self.government.xi_a, self.Gt_prob = self.action_wrapper(self.valid_action_dict[self.government.name])
        self.government.xi *= self.xi_max
        self.government.xi_a *= self.xi_max
        self.government.tau *= self.tau_max
        self.government.tau_a *= self.tau_max
        self.Gt_prob *= 0.3
        if np.mean(self.action_wrapper(self.valid_action_dict[self.government.name])) == 0:
            self.consumption_tax_rate = 0

        # households action
        multi_actions = self.action_wrapper(self.valid_action_dict[self.households.name])
        # saving_p = np.array(multi_actions[:, 0])[:,np.newaxis,...] * 0.3 + 0.7
        saving_p = np.array(multi_actions[:, 0])[:,np.newaxis,...]* 0.5 + 0.5
        self.saving_p = saving_p
        self.workingHours = np.array(multi_actions[:, 1])[:,np.newaxis,...]
        self.ht = self.workinghours_wrapper(self.workingHours)

        self.mean_saving_p = saving_p.mean()
        # market clear
        self.MarketClear()
        self.GDP = self.generate_gdp()
        # self.GDP - (self.alpha * (self.Kt/self.Lt)**(self.alpha-1) * self.Kt + self.WageRate * self.Lt)
        Gt = self.Gt_prob * self.GDP

        self.income = self.WageRate * self.households.e * self.ht + self.interest_rate * self.households.at
        self.income_tax, self.asset_tax = self.tax_function(self.income, self.households.at)

        self.post_income = self.income - self.income_tax
        post_asset = self.households.at - self.asset_tax
        total_wealth = self.post_income + post_asset

        # compute tax
        aggregate_consumption = (1 - saving_p) * total_wealth
        choose_consumption = 1/(1 + self.consumption_tax_rate) * aggregate_consumption

        if np.sum(choose_consumption) + Gt > self.GDP:
            self.done = True
        self.consumption = choose_consumption

        consumption_tax = self.consumption * self.consumption_tax_rate
        self.households.at_next = total_wealth - self.consumption * (1+self.consumption_tax_rate)
        self.tax_array = self.income_tax + self.asset_tax + consumption_tax
        self.Bt_next = (1 + self.interest_rate) * self.Bt + Gt - np.sum(self.tax_array)
        #self.Kt_next = np.sum(self.households.at_next) - self.Bt_next
        self.Kt_next = np.sum(self.households.at_next) - self.Bt_next + (self.alpha * (self.Kt/self.Lt)**(self.alpha-1)+1-self.depreciation_rate ) *self.Kt + (1+self.interest_rate) *(self.Bt - np.sum(self.households.at))

        # next state
        next_global_state, next_private_state = self.get_obs()
        # terminal
        self.wealth_gini = self.gini_coef(self.households.at_next)
        self.income_gini = self.gini_coef(self.post_income)

        # reward
        self.households_reward = self.utility_function(self.consumption, self.ht)
        self.government_reward = self.gov_reward()
        # terminal conditions: if gini>0.9; someone goes bankrupt; production capital < 0
        self.done = self.done or bool(self.wealth_gini > 0.9 or self.income_gini>0.9 or math.isnan(self.government_reward *self.wealth_gini * self.income_gini) or
                                      math.isnan(np.mean(self.households_reward)) or np.min(self.households.at_next) < 0 or self.Kt_next < 0)

        if math.isnan(self.government_reward * self.wealth_gini * self.income_gini) or math.isnan(np.mean(self.households_reward)):
            self.done = True
            self.ht = self.workinghours_wrapper(np.ones((self.households.n_households, 1)))
            self.consumption = np.zeros((self.households.n_households, 1)) + 0.001
            self.households_reward = self.utility_function(self.consumption, self.ht)
            self.government_reward = self.gov_reward()
            self.wealth_gini = 0.9
            self.income_gini = 0.9
            next_global_state = np.zeros(self.government.observation_space.shape[0])
            next_private_state = np.zeros((self.households.n_households, self.households.observation_space.shape[0]-self.government.observation_space.shape[0]))

        self.step_cnt += 1
        self.done = self.is_terminal()
        return next_global_state, next_private_state, self.government_reward, self.households_reward, self.done

    def gov_reward(self):
        '''capital GDP growth rate - weight * gini'''
        gov_goal = self.gov_task
        # gov_goal = "gini"
        if gov_goal == "gdp":
            return (self.per_household_gdp - self.old_per_gdp) / self.old_per_gdp
           
        elif gov_goal == "gini":
            return - (self.wealth_gini * self.income_gini)
        elif gov_goal == "social_welfare":
            return np.sum(self.households_reward)
        elif gov_goal == "gdp_gini":
            return (self.per_household_gdp - self.old_per_gdp) / self.old_per_gdp - self.gini_weight * (self.wealth_gini * self.income_gini)

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
        self.step_cnt = 0
        self.government.reset()
        self.households.reset()
        self.Kt_next = np.sum(self.households.at_next) * 0.5
        self.Bt_next = np.sum(self.households.at_next) * 0.5
        self.done = False

        self.Kt = copy.copy(self.Kt_next)
        self.Lt = np.sum(self.households.at) * (1.0-6.6*0.04)/6.6  # for calibration
        self.WageRate = (1 - self.alpha) * np.power(self.Kt / self.Lt, self.alpha)
        self.workingHours = np.ones((self.households.n_households,1))/3
        self.hours_max = self.Lt/np.sum(self.households.e * self.WageRate*self.workingHours)
        self.ht = self.workinghours_wrapper(self.workingHours)
        self.GDP = self.generate_gdp()
        self.income = self.households.e * self.ht

        self.display_mode = False
        return self.get_obs()

    def get_obs(self):
        '''
        v0:
            global state: top 10 mean income, top 10 mean asset, top 10 mean e, bottom 50 mean income, bottom 50 mean asset, bottom 50 mean e, wagerate
            private state: e, a
        '''
        income = self.income
        wealth = self.households.at_next
        sorted_income_index = sorted(range(len(income)), key=lambda k: income[k], reverse=True)
        top10_income_index = sorted_income_index[:int(0.1*self.households.n_households)]
        bottom50_income_index = sorted_income_index[int(0.5*self.households.n_households):]
        top10_income = income[top10_income_index]
        top10_e = self.households.e[top10_income_index]
        bot50_income = income[bottom50_income_index]
        bot50_e = self.households.e[bottom50_income_index]

        sorted_wealth_index = sorted(range(len(wealth)), key=lambda k: wealth[k], reverse=True)
        top10_wealth_index = sorted_wealth_index[:int(0.1*self.households.n_households)]
        bottom50_wealth_index = sorted_wealth_index[int(0.5*self.households.n_households):]
        top10_wealth = wealth[top10_wealth_index]
        bot50_wealth = wealth[bottom50_wealth_index]

        global_obs = np.array([np.mean(top10_wealth), np.mean(top10_income), np.mean(top10_e), np.mean(bot50_wealth), np.mean(bot50_income), np.mean(bot50_e), self.WageRate])
        private_obs = np.concatenate((self.households.e, wealth), -1)
        return global_obs, private_obs

    def utility_function(self, c_t, h_t):
        # life-time CRRA utility
        if 1-self.households.CRRA == 0:
            u_c = np.log(c_t/1000)  # u_c \in (9,20)
        else:
            u_c = c_t ** (1 - self.households.CRRA) / (1 - self.households.CRRA)
        if 1 + self.households.IFE == 0:
            u_h = np.log(h_t)
        else:
            u_h = 10*((self.workingHours)**(1 + self.households.IFE)/(1 + self.households.IFE))
        current_utility = u_c - u_h
        return current_utility

    def gini_coef(self, wealths):
        cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
        sum_wealths = cum_wealths[-1]
        xarray = np.array(range(0, len(cum_wealths))) / (len(cum_wealths) - 1)
        yarray = cum_wealths / sum_wealths
        B = np.trapz(yarray, x=xarray)
        A = 0.5 - B
        gini_coef = A / (A + B)
    
        return gini_coef

    def stacked_data(self, wealths):
        sorted_wealth = np.sort(wealths,axis=0)/np.sum(wealths)
        top_10 = sorted_wealth[int(0.9*self.households.n_households):].sum()
        top_20 = sorted_wealth[int(0.8*self.households.n_households):int(0.9*self.households.n_households)].sum()
        top_30 = sorted_wealth[int(0.7*self.households.n_households):int(0.8*self.households.n_households)].sum()
        top_40 = sorted_wealth[int(0.6*self.households.n_households):int(0.7*self.households.n_households)].sum()
        top_50 = sorted_wealth[int(0.5*self.households.n_households):int(0.6*self.households.n_households)].sum()
        top_60 = sorted_wealth[int(0.4*self.households.n_households):int(0.5*self.households.n_households)].sum()
        top_70 = sorted_wealth[int(0.3*self.households.n_households):int(0.4*self.households.n_households)].sum()
        top_80 = sorted_wealth[int(0.2*self.households.n_households):int(0.3*self.households.n_households)].sum()
        top_90 = sorted_wealth[int(0.1*self.households.n_households):int(0.2*self.households.n_households)].sum()
        top_100 = sorted_wealth[int(0*self.households.n_households):int(0.1*self.households.n_households)].sum()
        
        return [top_10, top_20, top_30, top_40, top_50, top_60, top_70, top_80, top_90, top_100]


    def tax_function(self, income, asset):
        # x: input
        def tax_f(x, tau, xi):
            return x - (1 - tau)/(1-xi) * np.power(x, 1-xi)

        income_tax = tax_f(income, self.government.tau, self.government.xi)
        asset_tax = tax_f(asset, self.government.tau_a, self.government.xi_a)
        return income_tax, asset_tax


    def _load_image(self):
        self.gov_img = pygame.image.load(os.path.join(ROOT_PATH, "img/gov.jpeg"))
        self.house_img = pygame.image.load(os.path.join(ROOT_PATH, "img/household.png"))
        self.firm_img = pygame.image.load(os.path.join(ROOT_PATH, "img/firm.png"))
        self.bank_img = pygame.image.load(os.path.join(ROOT_PATH, "img/bank.jpeg"))


    def render(self):
        if not self.display_mode:
            self.background = pygame.display.set_mode([500,500])
            self.display_mode = True
            self._load_image()

        self.background.fill((255,255,255))

        debug(f"Step {self.step_cnt}")
        debug("Mean Social Welfare: "+"{:.3g}".format(float(self.households_reward.mean())), x=280, y=10)
        debug("Wealth Gini: "+"{:.3g}".format(self.wealth_gini), x=348, y=30)
        debug("Income Gini: "+"{:.3g}".format(self.income_gini), x=348, y=50)
        debug('GDP: '+"{:.3g}".format(self.GDP), x=390, y=70)

        gov_img = pygame.transform.scale(self.gov_img,(50, 50))
        self.background.blit(gov_img, [100,100])
        debug("Tau: "+"{:.3g}".format(self.government.tau), x=10, y=80)
        debug("Xi: "+"{:.3g}".format(self.government.xi), x=10, y=100)
        debug("Tau_a"+"{:.3g}".format(self.government.tau_a), x=10, y=120)
        debug("Xi_a: "+"{:.3g}".format(self.government.xi_a), x=10, y=140)
        debug("Gt_prob: "+"{:.3g}".format(self.Gt_prob), x=10, y=160)
        debug("Bt2At: "+"{:.3g}".format(self.Bt2At), x=10, y=180)

        house_img = pygame.transform.scale(self.house_img, (50, 50))
        self.background.blit(house_img, [200,400])
        self.background.blit(house_img, [160,400])
        self.background.blit(house_img, [180,440])
        debug("Mean Working Hours: "+"{:.3g}".format(self.workingHours.mean()), x=250, y=450)
        debug("Mean Saving Prop: "+"{:.3g}".format(self.mean_saving_p), x=250, y=470)

        firm_img = pygame.transform.scale(self.firm_img, (50,50))
        self.background.blit(firm_img, [400,170])
        debug("Wage Rate: "+"{:.3g}".format(self.WageRate), x=370, y=230)

        pygame.draw.line(self.background, COLORS['blue'], (140,150), (190, 390), width=10)
        pygame.draw.line(self.background, COLORS['blue'], (220,390), (390, 210), width=10)
        pygame.draw.line(self.background, COLORS['blue'], (160,130), (390, 180), width=10)

        bank_img = pygame.transform.scale(self.bank_img, (50,50))
        self.background.blit(bank_img, [230,200])

        pygame.draw.line(self.background, COLORS['blue'], (145,145), (225, 195), width=10)
        pygame.draw.line(self.background, COLORS['blue'], (205,390), (250, 255), width=10)
        pygame.draw.line(self.background, COLORS['blue'], (380,195), (280, 230), width=10)

        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()





    def close(self):
        # 待修改
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

render=False
if render:


    COLORS = {
        'red': [255,0,0],
        'light red': [255, 127, 127],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'orange': [255, 127, 0],
        'grey':  [176,196,222],
        'purple': [160, 32, 240],
        'black': [0, 0, 0],
        'white': [255, 255, 255],
        'light green': [204, 255, 229],
        'sky blue': [0,191,255],
        # 'red-2': [215,80,83],
        # 'blue-2': [73,141,247]
    }


    pygame.init()
    font = pygame.font.Font(None, 22)
    def debug(info, y = 10, x=10, c='black'):
        display_surf = pygame.display.get_surface()
        debug_surf = font.render(str(info), True, COLORS[c])
        debug_rect = debug_surf.get_rect(topleft = (x,y))
        display_surf.blit(debug_surf, debug_rect)

