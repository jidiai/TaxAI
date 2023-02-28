from entities.base import BaseEntity
from utils.episode import EpisodeKey

import copy

class Government(BaseEntity):
    name='government'

    def __init__(self, n_agent, **entity_args):
        super().__init__(n_agent)


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