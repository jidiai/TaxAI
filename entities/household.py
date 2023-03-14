from entities.base import BaseEntity
from entities.government import Government
from utils.episode import EpisodeKey
import numpy as np
import copy
import math
import random

from gym.spaces import Box

class Household(BaseEntity):
    name='Household'

    #def __init__(self, **entity_args):
    def __init__(self, entity_args):
        super().__init__()

        self.n_households = entity_args.n_households
        # max action
        self.consumption_range = entity_args.consumption_range          #action range
        self.working_hours_range = entity_args.working_hours_range
        self.saving_range = entity_args.saving_prob                     #static constraint
        # fixed hyperparameter
        self.CRRA = entity_args.CRRA                                    #theta
        self.IFE = entity_args.IFE                                      #inverse Frisch Elasticity
        self.eta = 0                                                    # if eta=0, the transitory shocks are additive, if eta = 1, they are multiplicative
        self.beta = entity_args.beta                                    #discount factor


        # todo N households 初始化 e0, wealth0 ??? 看文献
        self.e0 = entity_args.initial_e                                 #e_0, initial abilities
        self.e = copy.deepcopy(self.e0)

        self.wealth = entity_args.initial_wealth

        # todo revise e generation
        # self.e_choice = entity_args.get('e_choice', [0.1, 2])
        # self.e_transition = eval(entity_args['e_transition'])         #lambda_1, lambda_2

        # obs government
        self.obs_tau = None
        self.obs_xi = None
        self.obs_tau_a = None
        self.obs_xi_a = None
        self.obs_G = None

        # space
        self.action_space = Box(
            low=-1, high=1, shape=(2,), dtype=np.float32  # todo low and high?
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )


    def e_transition(self, old_ep_index):
        '''
        已测试单人 e 计算
        '''
        et_elements = [-0.574, -0.232, 0.114, 0.133, 0.817, 1.245]
        et_prob = [0.263, 0.003, 0.556, 0.001, 0.001, 0.176]
        e_T = random.choices(et_elements, et_prob)[0]

        ep_elements = [0.580, 1.153, 1.926, 27.223]
        ep_prob = [[0.994, 0.002, 0.004, 0.00001],
                    [0.019, 0.979, 0.001, 9e-05],
                    [0.023, 0.000, 0.977, 5e-05],
                    [0.000, 0.000, 0.012, 0.987]]
        e_P = random.choices(ep_elements, ep_prob[old_ep_index])[0]


        e = e_P + e_T * math.pow(e_P, self.eta)
        return e

    def reset(self, **custom_cfg):
        # todo initial state

        self.WageRate = custom_cfg[EpisodeKey.WageRate]  #  todo custom cfg??
        self.saving_return = custom_cfg[EpisodeKey.SavingReturn]
        self.asset = custom_cfg[EpisodeKey.Asset]                       #update inside or outside?

    def get_obs(self, otherAgents):
        # todo add observation
        #{W_t, e_t, r_t-1, a_t, tau_t-1, xi_t-1, tau_{a, t-1}, xi_{a,t-1}, G_t-1}
        # 目前 other agents 只有政府
        government = otherAgents
        # rets = {
        #     EpisodeKey.WageRate: self.WageRate,
        #     EpisodeKey.Ability: self.e,
        #     EpisodeKey.SavingReturn: self.saving_return,
        #     EpisodeKey.Asset: self.asset,
        #     EpisodeKey.IncomeTax: government.tau,
        #     EpisodeKey.IncomeTaxSlope: government.xi,
        #     EpisodeKey.WealthTax: government.tau_a,
        #     EpisodeKey.WealthTaxSlope: government.xi_a,
        #     EpisodeKey.GovernmentSpending: government.G,
        #
        # }
        rets = np.concatenate(
            [
                self.WageRate,
                self.e,
                self.saving_return,
                self.asset,
                government.tau,
                government.xi,
                government.tau_a,
                government.xi_a,
                government.G,
            ]
        )

        return rets

    def get_actions(self):
        #if controllable, overwritten by the agent module
        pass

    def entity_step(self, action):


        # current state

        # todo next state
        self.state = 1

        # todo whether done?
        gini = 0
        step = 0

        terminated = bool(step > 1000 or gini > 0.7)


        # reward
        c_t = 0
        h_t = 0
        reward = self.utility_function(c_t,h_t)
        #abilitiy transition
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def utility_function(self, c_t, h_t):
        # todo 若输入矩阵，如何矩阵运算？
        # life-time CRRA utility
        if 1-self.CRRA == 0 or 1 + self.IFE == 0:
            print("Assignment error of CRRA or IFE!")
        current_utility = math.pow(c_t, 1-self.CRRA)/(1-self.CRRA) - math.pow(h_t, 1 + self.IFE)/(1 + self.IFE)
        return current_utility


    def render(self):
        # todo visualization
        pass

    def close(self):
        # 是否需要？
        pass




