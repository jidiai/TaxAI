from entities.base import BaseEntity
from utils.episode import EpisodeKey

import copy

class Firm(BaseEntity):
    name='Firm'

    def __init__(self, n_agent, **entity_args):
        super().__init__(n_agent)

        self.capital = entity_args[EpisodeKey.Capital]
        self.labor = entity_args[EpisodeKey.Labor]

    def reset(self, **custom_cfg):
        pass


    def get_obs(self):
        pass


    def get_actions(self):
        #if controllable, overwritten by the agent module
        pass

    def entity_step(self):
        #abilitiy transition
        pass
