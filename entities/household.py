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
        self.n_households = entity_args.n_households
        # max action
        self.consumption_range = entity_args.consumption_range          #action range
        self.working_hours_range = entity_args.working_hours_range

        # fixed hyperparameter
        self.CRRA = entity_args.CRRA                                    #theta
        self.IFE = entity_args.IFE                                      #inverse Frisch Elasticity
        self.eta = 0                                                    # if eta=0, the transitory shocks are additive, if eta = 1, they are multiplicative
        self.beta = entity_args.beta                                    #discount factor
        self.transfer = entity_args.lump_sum_transfer
        # todo N households 初始化 e0, wealth0 ??? 看文献
        # self.ep_index = entity_args.initial_e                                 #ep_0, initial abilities
        # self.e = self.e_transition(self.ep_index)

        self.e = self.e_distribution()
        self.asset = self.initial_wealth_distribution()
        self.next_asset = None   # 为了将二者区分开来

        # space
        self.action_space = Box(
            low=-1, high=1, shape=(2,), dtype=np.float32  # todo low and high?
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )


    def e_distribution(self):
        # todo 待修改与research papers对齐
        # 目前建模成人的能力是一个正态分布 智商在 normal(100，15)
        e0 = np.random.normal(loc=1, scale=0.15, size=[self.n_households,1])
        return e0
        # return torch.tensor(e0).unsqueeze(1)

    def e_transition(self, old_ep_index):
        '''
        已测试单人 e 计算, 目前 fix住e，没有转移 2023.3.14
        '''
        et_elements = [-0.574, -0.232, 0.114, 0.133, 0.817, 1.245]
        et_prob = [0.263, 0.003, 0.556, 0.001, 0.001, 0.176]
        e_T = random.choices(et_elements, et_prob)[0]

        ep_elements = [0.580, 1.153, 1.926, 27.223]
        ep_prob = np.array([[0.994, 0.002, 0.004, 0.00001],
                    [0.019, 0.979, 0.001, 9e-05],
                    [0.023, 0.000, 0.977, 5e-05],
                    [0.000, 0.000, 0.012, 0.987]])
        self.ep_index = random.choices(list(range(len(ep_elements))), ep_prob[old_ep_index])[0]
        e_P = ep_elements[self.ep_index]


        e = e_P + e_T * math.pow(e_P, self.eta)
        return e  # 换成tensor

    def reset(self, **custom_cfg):
        self.e = self.e_distribution()
        self.asset = self.initial_wealth_distribution()


    def get_obs(self, env):
        #{W_t, e_t, r_t-1, a_t, tau_t-1, xi_t-1, tau_{a, t-1}, xi_{a,t-1}, G_t-1}
        single_obs = np.array([env.WageRate,
                               env.RentRate,
                               env.government.tau,
                               env.government.xi,
                               env.government.tau_a,
                               env.government.xi_a,
                               env.government.G])

        multi_obs = np.repeat(single_obs[np.newaxis, ...], self.n_households, axis=0)
        multi_obs = np.concatenate((self.e, self.asset, multi_obs), -1)

        # todo 该格式怎么用？
        rets = {
            EpisodeKey.WageRate: env.WageRate,
            EpisodeKey.Ability: self.e,
            EpisodeKey.SavingReturn: env.RentRate,
            EpisodeKey.Asset: self.asset,
            EpisodeKey.IncomeTax: env.government.tau,
            EpisodeKey.IncomeTaxSlope: env.government.xi,
            EpisodeKey.WealthTax: env.government.tau_a,
            EpisodeKey.WealthTaxSlope: env.government.xi_a,
            EpisodeKey.GovernmentSpending: env.government.G,

        }

        return multi_obs

    def get_actions(self):
        #if controllable, overwritten by the agent module
        pass

    def entity_step(self, env, multi_actions=None):
        '''
        multi_actions = np.array([[p1,h1], [p2,h2],...,[pN, hN]]) (100 * 2)
        e.g.
        '''
        # multi_actions = np.random.random(size=(100, 2))

        saving_p = np.array(multi_actions[:, 0])[:,np.newaxis,...]
        self.workingHours = np.array(multi_actions[:, 1])[:,np.newaxis,...]

        self.income = env.WageRate * self.e * self.workingHours + env.RentRate * self.asset
        income_tax = env.government.tax_function(env.government.tau, env.government.xi, self.income)
        asset_tax = env.government.tax_function(env.government.tau, env.government.xi_a, self.asset)
        current_total_wealth = self.income - income_tax + self.asset - asset_tax + self.transfer

        # compute tax
        self.tax_array = income_tax + asset_tax - self.transfer  # N households tax array

        self.next_asset = saving_p * current_total_wealth
        current_consumption = (1 - saving_p) * current_total_wealth
        # self.e = self.e_transition(self.ep_index)  # 注意 e 有没有变

        self.reward = self.utility_function(current_consumption, self.workingHours)
        terminated = bool(self.gini_coef(self.next_asset) > 0.8)

        self.asset = copy.copy(self.next_asset)
        self.state = self.get_obs(env)
        return np.array(self.state, dtype=np.float32), self.reward, terminated

    def utility_function(self, c_t, h_t):
        # life-time CRRA utility
        if 1-self.CRRA == 0 or 1 + self.IFE == 0:
            print("Assignment error of CRRA or IFE!")
        current_utility = c_t**(1-self.CRRA)/(1-self.CRRA) - h_t**(1 + self.IFE)/(1 + self.IFE)
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
            a = 1  # pareto tail index, a 越大, 贫富差距越小 a=1, Gini=0.58
            return np.power(x, -1 / a)

        y = pareto(x)

        return np.array(y)[:,np.newaxis, ...]


    def render(self):
        # todo visualization
        pass

    def close(self):
        # 是否需要？
        pass




