from entities.base import BaseEntity
from entities.government import Government
from utils.episode import EpisodeKey
import numpy as np
import copy
import math
import random
import quantecon as qe
import matplotlib.pyplot as plt

from gym.spaces import Box

class Household(BaseEntity):
    name='Household'

    def __init__(self, entity_args):
        super().__init__()
        self.n_households = entity_args['n']
        # max action
        self.consumption_range = entity_args['consumption_range']          #action range
        self.working_hours_range = entity_args['working_hours_range']

        # fixed hyperparameter
        self.CRRA = entity_args['CRRA']                                    #theta
        self.IFE = entity_args['IFE']                                      #inverse Frisch Elasticity
        self.eta = entity_args['eta']                                       # if eta=0, the transitory shocks are additive, if eta = 1, they are multiplicative
        self.e_p = entity_args['e_p']
        self.e_q = entity_args['e_q']
        self.rho_e = entity_args['rho_e']
        self.sigma_e = entity_args['sigma_e']
        self.super_e = entity_args['super_e']

        # self.beta = entity_args.beta                                    #discount factor
        # self.transfer = entity_args['lump_sum_transfer']
        self.action_dim = entity_args['action_shape']
        # self.WageRate = entity_args['WageRate']
        # self.RentRate = entity_args['RentRate']

        self.reset()
        self.action_space = Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )


    # def e_distribution(self):
    #     # # todo 待修改与research papers对齐
    #     # # 目前建模成人的能力是一个正态分布 智商在 normal(100，15)
    #     # e0 = np.random.normal(loc=1, scale=0.15, size=[self.n_households,1])
    #     # return e0
    #     # # return torch.tensor(e0).unsqueeze(1)
    #     x = np.linspace(0.01, 1, self.n_households)
    #
    #     def pareto(x):
    #         a = 0.95  # pareto tail index, a 越大, e差距越小 a=0.95, Gini=0.65
    #         return np.power(x, -1 / a)
    #
    #     y = pareto(x)
    #
    #     return np.array(y)[:,np.newaxis, ...]

    def e_initial(self, n):
        self.e_array = np.zeros((n, 2))  # super-star and normal
        # 全部初始化为normal
        random_set = np.random.rand(n)
        self.e_array[:, 0] = (random_set > self.e_p).astype(int)
        self.e_array[:, 1] = (random_set < self.e_p).astype(int)
        self.e = np.sum(self.e_array, axis=1, keepdims=True)


    def generate_e_ability(self):
        """
        Generates n current ability levels for a given time step t.
        """
        self.e_past = copy.copy(self.e_array)
        e_past_mean = sum(self.e_past[:,0])/np.count_nonzero(self.e_past[:,0])
        for i in range(self.n_households):
            is_superstar = (self.e_array[i,1]>0).astype(int)
            if is_superstar == 0:
                # normal state
                if np.random.rand() < self.e_p:
                    # transit from normal to super-star
                    self.e_array[i, 0] = 0
                    self.e_array[i, 1] = self.super_e * e_past_mean
                else:
                    # remain in normal
                    self.e_array[i, 1] = 0
                    self.e_array[i, 0] = np.exp(self.rho_e * np.log(self.e_past[i, 0]) + self.sigma_e * np.random.randn())
            else:
                # super state
                if np.random.rand() < self.e_q:
                    # remain in super-star
                    self.e_array[i, 0] = 0
                    self.e_array[i, 1] = self.super_e * e_past_mean
                else:
                    # transit to normal
                    self.e_array[i, 1] = 0
                    self.e_array[i, 0] = random.uniform(self.e_array[:,0].min(), self.e_array[:,0].max())  # 随机来一个
        self.e = np.sum(self.e_array, axis=1, keepdims=True)


    # def e_transition(self, old_ep_index):
    #     '''
    #     已测试单人 e 计算, 目前 fix住e，没有转移 2023.3.14
    #     '''
    #     et_elements = [-0.574, -0.232, 0.114, 0.133, 0.817, 1.245]
    #     et_prob = [0.263, 0.003, 0.556, 0.001, 0.001, 0.176]
    #     e_T = random.choices(et_elements, et_prob)[0]
    #
    #     ep_elements = [0.580, 1.153, 1.926, 27.223]
    #     ep_prob = np.array([[0.994, 0.002, 0.004, 0.00001],
    #                 [0.019, 0.979, 0.001, 9e-05],
    #                 [0.023, 0.000, 0.977, 5e-05],
    #                 [0.000, 0.000, 0.012, 0.987]])
    #     self.ep_index = random.choices(list(range(len(ep_elements))), ep_prob[old_ep_index])[0]
    #     e_P = ep_elements[self.ep_index]
    #
    #
    #     e = e_P + e_T * math.pow(e_P, self.eta)
    #     return e  # 换成tensor

    def reset(self, **custom_cfg):
        # self.e = self.e_distribution()

        self.e_initial(self.n_households)
        self.generate_e_ability()
        self.at = self.initial_wealth_distribution()
        self.at_next = copy.copy(self.at)



    #
    # def get_obs(self):
    #     return np.concatenate((self.e, self.asset), -1)
    #
    # def get_actions(self):
    #     #if controllable, overwritten by the agent module
    #     pass

    # def entity_step(self, env, multi_actions=None):
    #     '''
    #     multi_actions = np.array([[p1,h1], [p2,h2],...,[pN, hN]]) (100 * 2)
    #     e.g.
    #     multi_actions = np.random.random(size=(100, 2))
    #     '''
    #     self.asset = copy.copy(self.next_asset)
    #     # multi_actions = np.random.random(size=(100, 2))
    #     saving_p = np.array(multi_actions[:, 0])[:,np.newaxis,...]
    #     self.workingHours = np.array(multi_actions[:, 1])[:,np.newaxis,...]
    #
    #     self.income = env.WageRate * self.e * self.workingHours + env.InterestRate * self.asset
    #     income_tax, asset_tax = env.government.tax_function(self.income, self.asset)
    #
    #     post_income = self.income - income_tax + self.transfer
    #     post_asset = self.asset - asset_tax
    #     current_total_wealth = post_income + post_asset
    #
    #     # compute tax
    #     self.tax_array = income_tax + asset_tax - self.transfer  # N households tax array
    #     self.consumption = (1 - saving_p) * current_total_wealth
    #     # self.e = self.e_transition(self.ep_index)  # 注意 e 有没有变
    #     # todo Yt> Ct + Gt
    #     Yt = env.generate_gdp()
    #     if Yt < (np.sum(self.consumption) + env.government.G):
    #         scale_p = Yt/(np.sum(self.consumption) + env.government.G)
    #         self.consumption = self.consumption * scale_p
    #         new_G = env.government.G * scale_p
    #         env.government.next_debt = env.government.next_debt + (new_G - env.government.G) * 10
    #     self.next_asset = current_total_wealth - self.consumption
    #
    #     self.reward = self.utility_function(self.consumption, self.workingHours)
    #     self.wealth_gini = self.gini_coef(current_total_wealth)
    #     self.income_gini = self.gini_coef(post_income)
    #     terminated = bool(self.wealth_gini > 0.9)
    #
    #     return self.reward, terminated

    # def utility_function(self, c_t, h_t):
    #     # life-time CRRA utility
    #     if 1-self.CRRA == 0:
    #         u_c = np.log(c_t)
    #     else:
    #         u_c = c_t ** (1 - self.CRRA) / (1 - self.CRRA)
    #     if 1 + self.IFE == 0:
    #         u_h = np.log(h_t)
    #     else:
    #         u_h = h_t**(1 + self.IFE)/(1 + self.IFE)
    #     current_utility = u_c - u_h
    #     return current_utility
    #
    # def gini_coef(self, wealths):
    #     '''
    #     cite: https://github.com/stephenhky/econ_inequality/blob/master/ginicoef.py
    #     '''
    #     cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    #     sum_wealths = cum_wealths[-1]
    #     xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths) - 1)
    #     yarray = cum_wealths / sum_wealths
    #     B = np.trapz(yarray, x=xarray)
    #     A = 0.5 - B
    #     return A / (A + B)

    def lorenz_curve(self, wealths):
        '''
        lorenz_curve: https://zhuanlan.zhihu.com/p/400411387
        '''
        f_vals, l_vals = qe.lorenz_curve(wealths)

        fig, ax = plt.subplots()
        ax.plot(f_vals, l_vals, label='Lorenz curve, lognormal sample')
        ax.plot(f_vals, f_vals, label='Lorenz curve, equality')
        ax.legend()
        plt.show()

    def initial_wealth_distribution(self):  # 大部分国家财富分布遵循 pareto distribution
        x = np.linspace(0.01, 1, self.n_households)

        def pareto(x):
            a = 0.68  # pareto tail index, a 越大, 贫富差距越小 a=0.68, Gini=0.85
            return np.power(x, -1 / a)

        y = pareto(x)

        return np.array(y)[:,np.newaxis, ...]


    def render(self):
        # todo visualization
        pass

    def close(self):
        # 是否需要？
        pass




