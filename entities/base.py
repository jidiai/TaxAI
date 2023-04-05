from abc import ABC, abstractmethod


class BaseEntity(ABC):
    name=""
    type=None
    # def __init__(self, n_agent):
    #     assert self.name
    #
    #     self.n_agent = n_agent

    def __init__(self):
        assert self.name


    def reset(self):
        pass

    # @abstractmethod
    # def get_obs(self):
    #     pass

    # @abstractmethod
    # def get_actions(self):
    #     pass
    #
    # @abstractmethod
    # def entity_step(self, action):
    #     pass


    def get_metrics(self):
        pass




