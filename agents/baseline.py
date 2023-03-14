import copy
import numpy as np
import torch
from torch import optim
import os,sys
sys.path.append(os.path.abspath('../..'))

from agents.models import flatten_mlp, tanh_gaussian_actor
from agents.log_path import make_logpath
from utils.experience_replay import replay_buffer

torch.autograd.set_detect_anomaly(True)

def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args
        # start to build the network.
        # todo initialize actor critic of government and households
        self.gov_actor = tanh_gaussian_actor(self.envs.observation_space.shape[0], self.envs.action_space.shape[0], self.args.hidden_size, \
                                            self.args.log_std_min, self.args.log_std_max)
        self.house_actor = tanh_gaussian_actor(self.envs.observation_space.shape[0], self.envs.action_space.shape[0], self.args.hidden_size, \
                                            self.args.log_std_min, self.args.log_std_max)

        # todo build up the network that will be used.
        self.gov_critic_1 = flatten_mlp(self.envs.observation_space.shape[0], self.args.hidden_size,
                               self.envs.action_space.shape[0])
        self.gov_critic_2 = flatten_mlp(self.envs.observation_space.shape[0], self.args.hidden_size,
                               self.envs.action_space.shape[0])
        # set the target q functions
        self.target_gov_qf1 = copy.deepcopy(self.gov_critic_1)
        self.target_gov_qf2 = copy.deepcopy(self.gov_critic_2)
        # todo households critic
        self.house_critic_1 = flatten_mlp(self.envs.observation_space.shape[0], self.args.hidden_size,
                               self.envs.action_space.shape[0])
        self.house_critic_2 = flatten_mlp(self.envs.observation_space.shape[0], self.args.hidden_size,
                               self.envs.action_space.shape[0])
        # set the target q functions
        self.target_house_qf1 = copy.deepcopy(self.house_critic_1)
        self.target_house_qf2 = copy.deepcopy(self.house_critic_2)

        # if use the cuda...
        if self.args.cuda:
            self.gov_actor.cuda()
            self.house_actor.cuda()
            self.gov_critic_1.cuda()
            self.gov_critic_2.cuda()
            self.house_critic_1.cuda()
            self.house_critic_2.cuda()

        # define the optimizer...
        self.gov_qf1_optim = torch.optim.Adam(self.gov_critic_1.parameters(), lr=self.args.q_lr)
        self.gov_qf2_optim = torch.optim.Adam(self.gov_critic_2.parameters(), lr=self.args.q_lr)
        self.house_qf1_optim = torch.optim.Adam(self.house_critic_2.parameters(), lr=self.args.q_lr)
        self.house_qf2_optim = torch.optim.Adam(self.house_critic_2.parameters(), lr=self.args.q_lr)
        # the optimizer for the policy network
        self.gov_actor_optim = torch.optim.Adam(self.gov_actor.parameters(), lr=self.args.p_lr)
        self.house_actor_optim = torch.optim.Adam(self.house_actor.parameters(), lr=self.args.p_lr)

        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)
        # get the action max
        self.action_max = self.envs.action_space.high[0]

        self.model_path, _ = make_logpath(self.args.env_name, "baseline")
        save_args(path=self.model_path, args=self.args)

        # get the observation
        # self.batch_ob_shape = (self.args.num_workers * self.args.nsteps, ) + self.envs.observation_space.shape
        # self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        # if self.args.env_type == 'mujoco':
        #     self.obs[:] = np.expand_dims(self.running_state(self.envs.reset()), 0)
        # else:
        #     self.obs[:] = self.envs.reset()
        # self.dones = [False for _ in range(self.args.num_workers)]
        # self.model_path, _ = make_logpath(self.args.env_name, "ppo", self.args.k)
        # save_args(path=self.model_path, args=self.args)


    def learn(self):
        pass