from entities.base import BaseEntity
from entities.household import Household
from entities.government import Government

class economic_society(BaseEntity):
    name = "wealth distribution economic society"
    def __init__(self, entity_args):
        super().__init__()
        self.ine = 1
        self.households = Household(entity_args)
        self.government = Government(entity_args)
        self.screen = None  # for rendering



    def reset(self, **custom_cfg):
        self.households.reset()
        self.government.reset()


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

