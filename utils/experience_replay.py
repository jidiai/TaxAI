import numpy as np
import random

"""
define the replay buffer and corresponding algorithms like PER

"""

class replay_buffer:
    def __init__(self, memory_size):
        self.storge = []
        self.memory_size = memory_size
        self.next_idx = 0

    # add the samples
    def add(self, global_obs, private_obs, gov_action, hou_action,  gov_reward, house_reward, next_global_obs, next_private_obs, done):
        data = (global_obs, private_obs, gov_action, hou_action,  gov_reward, house_reward, next_global_obs, next_private_obs, done)
        if self.next_idx >= len(self.storge):
            self.storge.append(data)
        else:
            self.storge[self.next_idx] = data
        # get the next idx
        self.next_idx = (self.next_idx + 1) % self.memory_size

    # encode samples
    def _encode_sample(self, idx):
        global_obses, private_obses, gov_actions, hou_actions, gov_rewards, house_rewards, next_global_obses, next_private_obses, dones = [], [], [], [], [], [], [], [], []
        for i in idx:
            data = self.storge[i]
            global_obs, private_obs, gov_action, hou_action,  gov_reward, house_reward, next_global_obs, next_private_obs, done = data
            global_obses.append(np.array(global_obs, copy=False))
            private_obses.append(np.array(private_obs, copy=False))
            gov_actions.append(np.array(gov_action, copy=False))
            hou_actions.append(np.array(hou_action, copy=False))
            gov_rewards.append(gov_reward)
            house_rewards.append(house_reward)
            next_global_obses.append(np.array(next_global_obs, copy=False))
            next_private_obses.append(np.array(next_private_obs, copy=False))
            dones.append(done)
        return np.array(global_obses), np.array(private_obses), np.array(gov_actions), np.array(hou_actions), np.array(gov_rewards), np.array(house_rewards),\
               np.array(next_global_obses), np.array(next_private_obses), np.array(dones)

    # sample from the memory
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storge) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


    # add the samples
    def mf_add(self, global_obs, private_obs, gov_action, hou_action, past_mean_action,  gov_reward, house_reward, next_global_obs, next_private_obs, done):
        data = (global_obs, private_obs, gov_action, hou_action, past_mean_action,  gov_reward, house_reward, next_global_obs, next_private_obs, done)
        if self.next_idx >= len(self.storge):
            self.storge.append(data)
        else:
            self.storge[self.next_idx] = data
        # get the next idx
        self.next_idx = int((self.next_idx + 1) % self.memory_size)

    # encode samples
    def mf_encode_sample(self, idx):
        global_obses, private_obses, gov_actions, hou_actions, past_mean_actions, gov_rewards, house_rewards, next_global_obses, next_private_obses, dones = [], [],[], [], [], [], [], [], [], []
        for i in idx:
            data = self.storge[i]
            global_obs, private_obs, gov_action, hou_action, past_mean_action,  gov_reward, house_reward, next_global_obs, next_private_obs, done = data
            global_obses.append(np.array(global_obs, copy=False))
            private_obses.append(np.array(private_obs, copy=False))
            gov_actions.append(np.array(gov_action, copy=False))
            hou_actions.append(np.array(hou_action, copy=False))
            past_mean_actions.append(np.array(past_mean_action, copy=False))
            gov_rewards.append(gov_reward)
            house_rewards.append(house_reward)
            next_global_obses.append(np.array(next_global_obs, copy=False))
            next_private_obses.append(np.array(next_private_obs, copy=False))
            dones.append(done)
        return np.array(global_obses), np.array(private_obses), np.array(gov_actions), np.array(hou_actions), np.array(past_mean_actions), np.array(gov_rewards), np.array(house_rewards),\
               np.array(next_global_obses), np.array(next_private_obses), np.array(dones)

    # sample from the memory
    def mf_sample(self, batch_size):
        idxes = [random.randint(0, len(self.storge) - 1) for _ in range(batch_size)]
        return self.mf_encode_sample(idxes)