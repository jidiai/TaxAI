import copy
import numpy as np
import torch
from torch import optim
import os,sys
sys.path.append(os.path.abspath('../..'))

from agents.models import SharedAgent, SharedCritic, Actor, Critic
from agents.log_path import make_logpath
from utils.experience_replay import replay_buffer
from agents.utils import get_action_info

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
        self.gov_actor = Actor(self.envs.government.observation_space.shape[0], self.envs.government.action_space.shape[0], self.args.hidden_size, \
                                            self.args.log_std_min, self.args.log_std_max)
        self.house_actor = SharedAgent(self.envs.households.observation_space.shape[0], self.envs.households.action_space.shape[0], self.args.n_households,
                                            self.args.log_std_min, self.args.log_std_max)

        self.gov_critic = Critic(self.envs.government.observation_space.shape[0], self.args.hidden_size, self.envs.government.action_space.shape[0])
        self.target_gov_qf = copy.deepcopy(self.gov_critic)
        self.house_critic = SharedCritic(self.envs.households.observation_space.shape[0], self.envs.government.action_space.shape[0],
                                         self.envs.households.action_space.shape[0], self.args.hidden_size)
        self.target_house_qf = copy.deepcopy(self.house_critic)


        # if use the cuda...
        if self.args.cuda:
            self.gov_actor.cuda()
            self.house_actor.cuda()
            self.gov_critic.cuda()
            self.house_critic.cuda()
            self.target_gov_qf.cuda()
            self.target_house_qf.cuda()

        # define the optimizer...
        self.gov_qf_optim = torch.optim.Adam(self.gov_critic.parameters(), lr=self.args.q_lr)
        self.house_qf_optim = torch.optim.Adam(self.house_critic.parameters(), lr=self.args.q_lr)
        # the optimizer for the policy network
        self.gov_actor_optim = torch.optim.Adam(self.gov_actor.parameters(), lr=self.args.p_lr)
        self.house_actor_optim = torch.optim.Adam(self.house_actor.parameters(), lr=self.args.p_lr)

        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)

        # get the action max
        self.gov_action_max = self.envs.government.action_space.high[0]
        self.hou_action_max = self.envs.households.action_space.high[0]

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
        # for loop
        global_timesteps = 0
        # before the official training, do the initial exploration to add episodes into the replay buffer
        self._initial_exploration(exploration_policy=self.args.init_exploration_policy)
        # reset the environment
        obs = self.envs.reset()
        rew = []
        epochs = []
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.train_loop_per_epoch):
                # for each epoch, it will reset the environment
                for t in range(self.args.epoch_length):
                    # start to collect samples
                    with torch.no_grad():
                        obs_tensor = self._get_tensor_inputs(obs)
                        pi = self.actor_net(obs_tensor)
                        action = get_action_info(pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                        action = action.cpu().numpy()[0]
                    # input the actions into the environment
                    obs_, reward, done, _ = self.env.step(self.action_max * action)
                    # store the samples
                    self.buffer.add(obs, action, reward, obs_, float(done))
                    # reassign the observations
                    obs = obs_
                    if done:
                        # reset the environment
                        obs = self.env.reset()
                # after collect the samples, start to update the network
                for _ in range(self.args.update_cycles):
                    qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss = self._update_newtork()
                    # update the target network
                    if global_timesteps % self.args.target_update_interval == 0:
                        self._update_target_network(self.target_qf1, self.qf1)
                        self._update_target_network(self.target_qf2, self.qf2)
                    global_timesteps += 1
            # print the log information
            if epoch % self.args.display_interval == 0:
                # start to do the evaluation
                mean_rewards = self._evaluate_agent()
                # store rewards and step
                now_step = (epoch + 1) * self.args.epoch_length
                rew.append(mean_rewards)
                np.savetxt(str(self.model_path) + "/reward.txt", rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)
                print(
                    '[{}] Epoch: {} / {}, Frames: {}, Rewards: {:.3f}, QF1: {:.3f}, QF2: {:.3f}, AL: {:.3f}, Alpha: {:.5f}, AlphaL: {:.5f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length, mean_rewards, qf1_loss,
                        qf2_loss, actor_loss, alpha, alpha_loss))
                # save models
                torch.save(self.actor_net.state_dict(), str(self.model_path) + '/model.pt')

        # government sample

        # save data into replay buffer

        # households sample
        # save data into replay buffer  # 提前初始化两个buffer

        # update

        # print
        # save rewards+steps +loss，  visualization

    # do the initial exploration by using the uniform policy
    def _initial_exploration(self, exploration_policy='gaussian'):
        # get the action information of the environment
        global_obs, private_obs = self.envs.reset()
        for _ in range(self.args.init_exploration_steps):
            if exploration_policy == 'uniform':
                raise NotImplementedError
            elif exploration_policy == 'gaussian':
                with torch.no_grad():
                    global_obs_tensor = self._get_tensor_inputs(global_obs)
                    private_obs_tensor = self._get_tensor_inputs(private_obs)
                    gov_pi = self.gov_actor(global_obs_tensor)
                    gov_action = get_action_info(gov_pi).select_actions(reparameterize=False)
                    hou_pi = self.house_actor(global_obs_tensor, private_obs_tensor, gov_action)
                    hou_action = get_action_info(hou_pi).select_actions(reparameterize=False)
                    gov_action = gov_action.cpu().numpy()[0]
                    hou_action = hou_action.cpu().numpy()[0]
                    action = {self.envs.government.name: self.gov_action_max * gov_action,
                              self.envs.households.name: self.hou_action_max * hou_action}
                    next_obs, gov_reward, house_reward, done = self.envs.step(action)

                # store the episodes
                self.buffer.add(obs, gov_action, hou_action,  gov_reward, house_reward, next_obs, float(done))

                obs = next_obs
                if done:
                    # if done, reset the environment
                    obs = self.envs.reset()
        print("Initial exploration has been finished!")

    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
        return obs_tensor

    def _update_newtork(self):
        # smaple batch of samples from the replay buffer
        obses, actions, rewards, obses_, dones = self.buffer.sample(self.args.batch_size)
        # preprocessing the data into the tensors, will support GPU later
        obses = torch.tensor(obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        actions = torch.tensor(actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        rewards = torch.tensor(rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        obses_ = torch.tensor(obses_, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - dones, dtype=torch.float32,
                                     device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        # start to update the actor network
        pis = self.actor_net(obses)
        actions_info = get_action_info(pis, cuda=self.args.cuda)
        actions_, pre_tanh_value = actions_info.select_actions(reparameterize=True)
        log_prob = actions_info.get_log_prob(actions_, pre_tanh_value)
        # use the automatically tuning
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        # get the param
        alpha = self.log_alpha.exp()

        # q value function loss
        q1_value = self.qf1(obses, actions)
        q2_value = self.qf2(obses, actions)
        with torch.no_grad():
            pis_next = self.actor_net(obses_)
            actions_info_next = get_action_info(pis_next, cuda=self.args.cuda)
            actions_next_, pre_tanh_value_next = actions_info_next.select_actions(reparameterize=True)
            log_prob_next = actions_info_next.get_log_prob(actions_next_, pre_tanh_value_next)
            target_q_value_next = torch.min(self.target_qf1(obses_, actions_next_),
                                            self.target_qf2(obses_, actions_next_)) - alpha * log_prob_next
            target_q_value = self.args.reward_scale * rewards + inverse_dones * self.args.gamma * target_q_value_next
        qf1_loss = (q1_value - target_q_value).pow(2).mean()
        qf2_loss = (q2_value - target_q_value).pow(2).mean()
        # qf1
        self.qf1_optim.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optim.step()
        # qf2
        self.qf2_optim.zero_grad()
        qf2_loss.backward()
        self.qf2_optim.step()

        # get the q_value for new actions
        q_actions_ = torch.min(self.qf1(obses, actions_), self.qf2(obses, actions_))
        actor_loss = (alpha * log_prob - q_actions_).mean()
        # policy loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha.item(), alpha_loss.item()

    # update the target network
    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    # evaluate the agent
    def _evaluate_agent(self):
        total_reward = 0
        for _ in range(self.args.eval_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0
            while True:
                with torch.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs)
                    pi = self.actor_net(obs_tensor)
                    action = get_action_info(pi, cuda=self.args.cuda).select_actions(exploration=False,
                                                                                     reparameterize=False)
                    action = action.detach().cpu().numpy()[0]
                # input the action into the environment
                obs_, reward, done, _ = self.eval_env.step(self.action_max * action)
                episode_reward += reward
                if done:
                    break
                obs = obs_
            total_reward += episode_reward
        return total_reward / self.args.eval_episodes
