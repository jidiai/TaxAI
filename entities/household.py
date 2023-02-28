from entities.base import BaseEntity
from utils.episode import EpisodeKey

import copy

class Household(BaseEntity):
    name='Household'

    def __init__(self, n_agent, **entity_args):
        super().__init__(n_agent)

        self.consumption_range = entity_args['consumption_range']       #action range
        self.working_hours_range = entity_args['working_hours_range']
        self.saving_range = entity_args['saving_range']                 #static constraint

        self.CRRA = entity_args['CRRA']                                 #theta
        self.IFE = entity_args['IFE']                                   #inverse Frisch Elasticity
        self.e0 = entity_args['initial_e']                              #e_0, initial abilities
        self.e_choice = entity_args.get('e_choice', [0.1, 2])
        self.e_transition = eval(entity_args['e_transition'])           #lambda_1, lambda_2
        self.beta = entity_args['beta']                                 #discount factor

        self.wealth = entity_args['initial_wealth']
        self.e = copy.deepcopy(self.e0)

    def reset(self, **custom_cfg):
        self.WageRate = custom_cfg[EpisodeKey.WageRate]
        self.saving_return = custom_cfg[EpisodeKey.SavingReturn]
        self.asset = custom_cfg[EpisodeKey.Asset]                       #update inside or outside?

    def get_obs(self):
        #{W_t, e_t, r_t-1, a_t, tau_t-1, xi_t-1, tau_a, xi_a,t-1}
        rets = {
            EpisodeKey.WageRate: self.WageRate,
            EpisodeKey.Ability: self.e,
            EpisodeKey.SavingReturn: self.saving_return,
            EpisodeKey.Asset: self.asset,
        }

        return rets

    def get_actions(self):
        #if controllable, overwritten by the agent module
        pass

    def entity_step(self):
        #abilitiy transition
        pass




