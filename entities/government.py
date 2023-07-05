from TaxAI.entities.base import BaseEntity
from TaxAI.utils.episode import EpisodeKey
import math
import copy
import numpy as np
from gym.spaces import Box

class Government(BaseEntity):
    name='government'

    def __init__(self, entity_args):
        super().__init__()
        self.entity_args = entity_args

        self.reset()
        self.action_dim = entity_args['action_shape']

        self.action_space = Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )


    def reset(self, **custom_cfg):
        pass




