import numpy as np
import torch
import torch.nn.functional as F

import os,sys
sys.path.append(os.path.abspath('../..'))
from agents.models import Actor, Critic
from datetime import datetime
from agents.log_path import make_logpath
# from mpi4py import MPI
from agents.utils import get_action_info
from utils.experience_replay import replay_buffer
from agents.utils import ounoise
import copy
import gym
import wandb

"""
ddpg algorithms - revised baseline version

support MPI training

"""
def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class independent_agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args
        self.eval_env = copy.copy(envs)
        # start to build the network.
        self.gov_actor = Actor(self.envs.government.observation_space.shape[0],
                               self.envs.government.action_space.shape[0], self.args.hidden_size, \
                               self.args.log_std_min, self.args.log_std_max)
        self.house_actor = Actor(self.envs.households.observation_space.shape[0],
                                       self.envs.households.action_space.shape[0], self.args.hidden_size,
                                       self.args.log_std_min, self.args.log_std_max)

        self.gov_critic = Critic(self.envs.government.observation_space.shape[0], self.args.hidden_size,
                                 self.envs.government.action_space.shape[0])
        self.house_critic = Critic(self.envs.households.observation_space.shape[0], self.args.hidden_size,
                                   self.envs.households.action_space.shape[0])

        self.gov_actor_target_net = copy.deepcopy(self.gov_actor)
        self.house_actor_target_net = copy.deepcopy(self.house_actor)
        self.gov_critic_target_net = copy.deepcopy(self.gov_critic)
        self.house_critic_target_net = copy.deepcopy(self.house_critic)

        # define the optimizer...
        self.gov_critic_optim = torch.optim.Adam(self.gov_critic.parameters(), lr=self.args.q_lr)
        self.house_critc_optim = torch.optim.Adam(self.house_critic.parameters(), lr=self.args.q_lr)
        # the optimizer for the policy network
        self.gov_actor_optim = torch.optim.Adam(self.gov_actor.parameters(), lr=self.args.p_lr)
        self.house_actor_optim = torch.optim.Adam(self.house_actor.parameters(), lr=self.args.p_lr)

        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)

        # get the action max
        self.gov_action_max = self.envs.government.action_space.high[0]
        self.hou_action_max = self.envs.households.action_space.high[0]

        # if use the cuda...
        if self.args.cuda:
            self.gov_actor.cuda()
            self.house_actor.cuda()
            self.gov_critic.cuda()
            self.house_critic.cuda()
            self.gov_actor_target_net.cuda()
            self.house_actor_target_net.cuda()
            self.gov_critic_target_net.cuda()
            self.house_critic_target_net.cuda()

        # create the noise generator
        self.gov_noise_generator = ounoise(std=0.2, action_dim=self.envs.government.action_dim)
        self.house_noise_generator = ounoise(std=0.2, action_dim=self.envs.households.action_dim)
        self.model_path, _ = make_logpath(algo="independent")
        save_args(path=self.model_path, args=self.args)
        wandb.init(
            config=self.args,
            project="AI_TaxingPolicy",
            entity="ai_tax",
            name=self.model_path.parent.name + "-"+ self.model_path.name + '  n=' + str(self.args.n_households),
            dir=str(self.model_path),
            job_type="training",
            reinit=True
        )


    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
        return obs_tensor

    def obs_concate(self, global_obs, private_obs, update=True):
        if update == True:
            global_obs = global_obs.unsqueeze(1)
        n_global_obs = global_obs.repeat(1, self.args.n_households, 1)
        return torch.cat([n_global_obs, private_obs], dim=-1).flatten(0,1)



    def learn(self):
        """
        the learning part

        """
        global_timesteps = 0
        self.gov_noise_generator.reset()
        self.house_noise_generator.reset()
        global_obs, private_obs = self.envs.reset()
        gov_rew = []
        house_rew = []
        epochs = []

        for epoch in range(self.args.n_epochs):
            for t in range(self.args.epoch_length):
                # start to collect samples
                global_obs_tensor = self._get_tensor_inputs(global_obs)
                private_obs_tensor = self._get_tensor_inputs(private_obs)
                gov_pi = self.gov_actor(global_obs_tensor)
                gov_action = get_action_info(gov_pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                hou_pi = self.house_actor(self.obs_concate(global_obs_tensor, private_obs_tensor, update=False))
                hou_action = get_action_info(hou_pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                gov_action = gov_action.cpu().numpy()[0]
                hou_action = hou_action.cpu().numpy()
                gov_action, hou_action = self.add_noise(gov_action, hou_action)
                action = {self.envs.government.name: gov_action,
                          self.envs.households.name: hou_action}
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)

                # store the episodes
                self.buffer.add(global_obs, private_obs, gov_action, hou_action, gov_reward, house_reward,
                                next_global_obs, next_private_obs, float(done))

                global_obs = next_global_obs
                private_obs = next_private_obs
                if done:
                    # if done, reset the environment
                    global_obs, private_obs = self.envs.reset()
                    self.gov_noise_generator.reset()
                    self.house_noise_generator.reset()
                # after collect the samples, start to update the network
            for _ in range(self.args.update_cycles):
                gov_actor_loss, gov_critic_loss, house_actor_loss, house_critic_loss = self._update_network()
                # update the target network
                if global_timesteps % self.args.target_update_interval == 0:
                    self._update_target_network(self.gov_critic_target_net, self.gov_critic)
                    self._update_target_network(self.house_critic_target_net, self.house_critic)
                    self._update_target_network(self.gov_actor_target_net, self.gov_actor)
                    self._update_target_network(self.house_actor_target_net, self.house_actor)

                global_timesteps += 1
                # print the log information
            if epoch % self.args.display_interval == 0:
                # start to do the evaluation
                mean_gov_rewards, mean_house_rewards = self._evaluate_agent()
                # store rewards and step
                now_step = (epoch + 1) * self.args.epoch_length
                gov_rew.append(mean_gov_rewards)
                house_rew.append(mean_house_rewards)
                np.savetxt(str(self.model_path) + "/gov_reward.txt", gov_rew)
                np.savetxt(str(self.model_path) + "/house_reward.txt", house_rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)

                # GDP + mean utility + wealth distribution + income distribution
                wandb.log({"mean households utility": mean_house_rewards,
                           "goverment utility": mean_gov_rewards,
                           "wealth gini": self.envs.households.wealth_gini,
                           "income gini": self.envs.households.income_gini,
                           "GDP": self.envs.GDP,
                           "government actor loss": gov_actor_loss,
                           "government critic loss": gov_critic_loss,
                           "households actor loss": house_actor_loss,
                           "households critic loss": house_critic_loss,
                           "steps": now_step})

                print(
                    '[{}] Epoch: {} / {}, Frames: {}, gov_Rewards: {:.3f}, house_Rewards: {:.3f}, gov_actor_loss: {:.3f}, gov_critic_loss: {:.3f}, house_actor_loss: {:.3f}, house_critic_loss: {:.3f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length,
                        mean_gov_rewards, mean_house_rewards, gov_actor_loss, gov_critic_loss, house_actor_loss,
                        house_critic_loss))
                # save models
                torch.save(self.gov_actor.state_dict(), str(self.model_path) + '/gov_actor.pt')
                torch.save(self.house_actor.state_dict(), str(self.model_path) + '/house_actor.pt')

        wandb.finish()

    # update the target network
    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    # this function will choose action for the agent and do the exploration
    def add_noise(self, gov_action, house_action):
        gov_action = gov_action + self.gov_noise_generator.noise()
        house_action = house_action + self.house_noise_generator.noise()
        gov_action = np.clip(gov_action, -1, 1)
        house_action = np.clip(house_action, -1, 1)
        return gov_action, house_action

    # update the network
    def _update_network(self):
        # smaple batch of samples from the replay buffer
        global_obses, private_obses, gov_actions, hou_actions, gov_rewards,\
        house_rewards, next_global_obses, next_private_obses, dones = self.buffer.sample(self.args.batch_size)
        # preprocessing the data into the tensors, will support GPU later
        global_obses = torch.tensor(global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        private_obses = torch.tensor(private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_actions = torch.tensor(gov_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        hou_actions = torch.tensor(hou_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_rewards = torch.tensor(gov_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        house_rewards = torch.tensor(house_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_global_obses = torch.tensor(next_global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_private_obses = torch.tensor(next_private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - dones, dtype=torch.float32,
                                     device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        n_inverse_dones = inverse_dones.repeat(self.args.n_households, 1)
        with torch.no_grad():
            next_gov_pi = self.gov_actor(next_global_obses)
            next_gov_action, _ = get_action_info(next_gov_pi, cuda=self.args.cuda).select_actions(reparameterize=True)
            gov_td_target = gov_rewards + inverse_dones * self.args.gamma * self.gov_critic_target_net(next_global_obses, next_gov_action)
            next_house_pi = self.house_actor(self.obs_concate(next_global_obses, next_private_obses))
            next_house_action, _ = get_action_info(next_house_pi, cuda=self.args.cuda).select_actions(reparameterize=True)
            house_td_target = house_rewards.flatten(0,1) + n_inverse_dones * self.args.gamma * self.house_critic_target_net(
                self.obs_concate(next_global_obses, next_private_obses), next_house_action)

        gov_q_value = self.gov_critic(global_obses, gov_actions)
        gov_critic_loss = torch.mean(F.mse_loss(gov_q_value, gov_td_target.detach()))

        house_q_value = self.house_critic(self.obs_concate(global_obses, private_obses), hou_actions.flatten(0,1))
        house_critic_loss = torch.mean(F.mse_loss(house_q_value, house_td_target.detach()))
        # the actor loss
        gov_pis = self.gov_actor(global_obses)
        gov_actions_, _ = get_action_info(gov_pis, cuda=self.args.cuda).select_actions(reparameterize=True)
        gov_actor_loss = -self.gov_critic(global_obses, gov_actions_).mean()

        house_pis = self.house_actor(self.obs_concate(global_obses, private_obses))
        house_actions_, _ = get_action_info(house_pis, cuda=self.args.cuda).select_actions(reparameterize=True)
        house_actor_loss = -self.house_critic(self.obs_concate(global_obses, private_obses), house_actions_).mean()


        self.gov_actor_optim.zero_grad()
        self.gov_critic_optim.zero_grad()
        self.house_actor_optim.zero_grad()
        self.house_critc_optim.zero_grad()
        gov_actor_loss.backward()
        gov_critic_loss.backward()
        house_actor_loss.backward()
        house_critic_loss.backward()
        self.gov_actor_optim.step()
        self.gov_critic_optim.step()
        self.house_actor_optim.step()
        self.house_critc_optim.step()

        return gov_actor_loss.item(), gov_critic_loss.item(), house_actor_loss.item(), house_critic_loss.item()


    # evaluate the agent
    def _evaluate_agent(self):
        total_gov_reward = 0
        total_house_reward = 0
        for _ in range(self.args.eval_episodes):
            global_obs, private_obs = self.eval_env.reset()
            episode_gov_reward = 0
            episode_mean_house_reward = 0
            while True:
                with torch.no_grad():
                    # start to collect samples
                    global_obs_tensor = self._get_tensor_inputs(global_obs)
                    private_obs_tensor = self._get_tensor_inputs(private_obs)
                    gov_pi = self.gov_actor(global_obs_tensor)
                    gov_action = get_action_info(gov_pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                    hou_pi = self.house_actor(self.obs_concate(global_obs_tensor, private_obs_tensor, update=False))
                    hou_action = get_action_info(hou_pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                    gov_action = gov_action.cpu().numpy()[0]
                    hou_action = hou_action.cpu().numpy()
                    gov_action, hou_action = self.add_noise(gov_action, hou_action)
                    action = {self.envs.government.name: gov_action,
                              self.envs.households.name: hou_action}
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)

                episode_gov_reward += gov_reward
                episode_mean_house_reward += np.mean(house_reward)
                if done:
                    break
                global_obs = next_global_obs
                private_obs = next_private_obs

            total_gov_reward += episode_gov_reward
            total_house_reward += episode_mean_house_reward
        avg_gov_reward = total_gov_reward / self.args.eval_episodes
        avg_house_reward = total_house_reward / self.args.eval_episodes
        return avg_gov_reward[0], avg_house_reward



