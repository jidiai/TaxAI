import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import os,sys
import wandb
sys.path.append(os.path.abspath('../..'))

from agents.models import Actor, MFCritic, BMF_actor, BMF_critic, BMF_actor_1
from agents.log_path import make_logpath
from utils.experience_replay import replay_buffer
from agents.utils import get_action_info
from datetime import datetime
from tensorboardX import SummaryWriter
from env.evaluation import save_parameters

torch.autograd.set_detect_anomaly(True)

def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

'''
Bi-level: government and households 
mean field: 
    - gov has a actor and a critic;  pi(og), Q(og, ag, bar{ah})
    - households share a actor and a critic. pi(at | ot, ag, bar{a})  Q(oh, ag, ah^i, bar{ah^-i} )
'''
class BMFAC_agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args
        self.eval_env = copy.copy(envs)
        # start to build the network.
        gov_obs_dim = self.envs.government.observation_space.shape[0]
        gov_action_dim = self.envs.government.action_space.shape[0]
        house_obs_dim = self.envs.households.observation_space.shape[0]
        house_action_dim = self.envs.households.action_space.shape[1]

        self.gov_actor = Actor(gov_obs_dim, gov_action_dim, self.args.hidden_size, self.args.log_std_min, self.args.log_std_max)
        # self.house_actor = BMF_actor(house_obs_dim, gov_action_dim, house_action_dim, self.args.n_households, self.args.log_std_min, self.args.log_std_max)
        self.house_actor = BMF_actor_1(house_obs_dim, gov_action_dim, house_action_dim, self.args.n_households, self.args.log_std_min, self.args.log_std_max)
        self.gov_critic = MFCritic(gov_obs_dim, self.args.hidden_size, gov_action_dim, house_action_dim*2)
        self.target_gov_qf = copy.deepcopy(self.gov_critic)
        self.house_critic = BMF_critic(house_obs_dim, gov_action_dim, house_action_dim, self.args.hidden_size, self.args.n_households)
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
        self.gov_critic_optim = torch.optim.Adam(self.gov_critic.parameters(), lr=self.args.q_lr)
        self.house_critc_optim = torch.optim.Adam(self.house_critic.parameters(), lr=self.args.q_lr)
        # the optimizer for the policy network
        self.gov_actor_optim = torch.optim.Adam(self.gov_actor.parameters(), lr=self.args.p_lr)
        self.house_actor_optim = torch.optim.Adam(self.house_actor.parameters(), lr=self.args.p_lr)

        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)

        # get the action max
        self.gov_action_max = self.envs.government.action_space.high[0]
        self.hou_action_max = self.envs.households.action_space.high[0][0]

        self.model_path, _ = make_logpath(algo="bmfac",n=self.args.n_households)
        save_args(path=self.model_path, args=self.args)
        self.fix_gov = True
        wandb.init(
            config=self.args,
            project="AI_TaxingPolicy",
            entity="ai_tax",
            name=self.model_path.parent.parent.name+ "-"+ self.model_path.name +'  n='+ str(self.args.n_households),
            dir=str(self.model_path),
            job_type="training",
            reinit=True
        )

    def house_action_initialize(self):
        action = np.random.uniform(low=-self.hou_action_max, high=self.hou_action_max, size=(1, self.args.n_households, self.envs.households.action_space.shape[1]))
        mean_action = np.mean(action, axis=1)
        return mean_action

    def multiple_households_mean_action(self, actions=None):
        # 根据wealth排序 前10% 的人动作平均 和 bottom 50%的action mean
        if actions is None:
            action = np.random.uniform(low=-self.hou_action_max, high=self.hou_action_max,
                                       size=(1, self.args.n_households, self.envs.households.action_space.shape[1]))
            mean_action = np.mean(action, axis=1)
            return np.hstack((mean_action, mean_action))
        else:
            wealth = self.envs.households.at_next
            sorted_wealth_index = sorted(range(len(wealth)), key=lambda k: wealth[k], reverse=True)
            top10_wealth_index = sorted_wealth_index[:int(0.1 * self.args.n_households)]
            bottom50_wealth_index = sorted_wealth_index[int(0.5 * self.args.n_households):]
            top10_action = actions[top10_wealth_index]
            bot50_action = actions[bottom50_wealth_index]
            # return top10_action, bot50_action
            return np.hstack((np.mean(top10_action,axis=0), np.mean(bot50_action,axis=0)))[np.newaxis,:]
    def get_tensor_mean_action(self, actions, wealth):
        sorted_wealth_index = torch.sort(wealth[:, :, 0], dim=1)[1]
        top10_wealth_index = sorted_wealth_index[:, :int(0.1 * self.args.n_households)]
        bottom50_wealth_index = sorted_wealth_index[:, int(0.5 * self.args.n_households):]
        top10_action = actions.gather(1, top10_wealth_index.unsqueeze(2).expand(-1, -1, self.envs.households.action_space.shape[1]))
        bot50_action = actions.gather(1, bottom50_wealth_index.unsqueeze(2).expand(-1, -1, self.envs.households.action_space.shape[1]))

        return torch.cat((torch.mean(top10_action, dim=1), torch.mean(bot50_action, dim=1)), 1)

    def learn(self):
        update_freq = self.args.update_freq
        initial_train = self.args.initial_train
        # for loop
        global_timesteps = 0
        # before the official training, do the initial exploration to add episodes into the replay buffer
        self._initial_exploration(exploration_policy=self.args.init_exploration_policy)
        # reset the environment
        global_obs, private_obs = self.envs.reset()
        past_mean_house_action = self.multiple_households_mean_action()
        gov_rew = []
        house_rew = []
        epochs = []
        wealth_stack = []
        income_stack = []
        agent_list = ["households", "government"]
        update_index = 0

        for epoch in range(self.args.n_epochs):
            if (epoch // update_freq) % 2 == 0 and epoch > initial_train:
                update_index = 1 - update_index
            update_agent = agent_list[update_index]

            # for each epoch, it will reset the environment
            for t in range(self.args.epoch_length):
                # start to collect samples
                global_obs_tensor = self._get_tensor_inputs(global_obs)
                private_obs_tensor = self._get_tensor_inputs(private_obs)
                gov_pi = self.gov_actor(global_obs_tensor)
                gov_action = get_action_info(gov_pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                hou_pi = self.house_actor(global_obs_tensor, private_obs_tensor, gov_action)
                hou_action = get_action_info(hou_pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                gov_action = gov_action.cpu().numpy()[0]
                hou_action = hou_action.cpu().numpy()[0]

                if epoch < initial_train:
                    self.fix_gov = True
                    gov_action = np.array([0.1/0.5, 0.0/0.05, 0, 0, 0.1]) + np.random.normal(0,0.1, size=(5,))
                else:
                    self.fix_gov = False

                past_mean_house_action = self.multiple_households_mean_action(hou_action)[0]
                action = {self.envs.government.name: gov_action,
                          self.envs.households.name: hou_action}
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)

                # store the episodes
                self.buffer.mf_add(global_obs, private_obs, gov_action, hou_action, past_mean_house_action, gov_reward, house_reward,
                                next_global_obs, next_private_obs, float(done))
                # past_mean_house_action = self.multiple_households_mean_action(hou_action)
                global_obs = next_global_obs
                private_obs = next_private_obs
                '''save variables'''
                # if epoch == 1 or epoch == 100 or epoch == 1000:
                #     if self.envs.step_cnt == 1 or self.envs.step_cnt == 100 or self.envs.step_cnt == 299:
                #         save_parameters(self.model_path, self.envs.step_cnt, epoch, self.envs)
                if done:
                    # if done, reset the environment
                    global_obs, private_obs = self.envs.reset()
            # after collect the samples, start to update the network
            for _ in range(self.args.update_cycles):
                gov_actor_loss, gov_critic_loss, house_actor_loss, house_critic_loss = self._update_network(update_agent=update_agent)
                # update the target network
                if global_timesteps % self.args.target_update_interval == 0:
                    self._update_target_network(self.target_gov_qf, self.gov_critic)
                    self._update_target_network(self.target_house_qf, self.house_critic)
                global_timesteps += 1
            # print the log information
            if epoch % self.args.display_interval == 0:
                # start to do the evaluation
                mean_gov_rewards, mean_house_rewards, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years, avg_wealth_stacked, avg_income_stacked = self._evaluate_agent()
                # store rewards and step
                now_step = (epoch + 1) * self.args.epoch_length
                gov_rew.append(mean_gov_rewards)
                house_rew.append(mean_house_rewards)
                np.savetxt(str(self.model_path) + "/gov_reward.txt", gov_rew)
                np.savetxt(str(self.model_path) + "/house_reward.txt", house_rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)
                np.savetxt(str(self.model_path) + "/wealth_stack.txt", wealth_stack)
                np.savetxt(str(self.model_path) + "/income_stack.txt", income_stack)

                wandb.log({"mean households utility": mean_house_rewards,
                           "goverment utility": mean_gov_rewards,
                           "years": years,
                           "wealth gini": avg_wealth_gini,
                           "income gini": avg_income_gini,
                           "GDP": avg_gdp,
                           "tax per households": avg_mean_tax,
                           "post income per households": avg_mean_post_income,
                           "wealth per households": avg_mean_wealth,
                           "government actor loss": gov_actor_loss,
                           "government critic loss": gov_critic_loss,
                           "households actor loss": house_actor_loss,
                           "households critic loss": house_critic_loss,
                           "steps": now_step})

                print(
                    '[{}] Epoch: {} / {}, Frames: {}, gov_Rewards: {:.3f}, house_Rewards: {:.3f}, years:{:.3f}, gov_actor_loss: {:.3f}, gov_critic_loss: {:.3f}, house_actor_loss: {:.3f}, house_critic_loss: {:.3f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length, mean_gov_rewards, mean_house_rewards,years, gov_actor_loss, gov_critic_loss, house_actor_loss, house_critic_loss))
                # save models
                torch.save(self.gov_actor.state_dict(), str(self.model_path) + '/gov_actor.pt')
                torch.save(self.house_actor.state_dict(), str(self.model_path) + '/house_actor.pt')

        wandb.finish()

    def test(self):
        self.gov_actor.load_state_dict(torch.load("/home/mqr/code/AI-TaxingPolicy/agents/models/bmfac/run105/gov_actor.pt"))
        self.house_actor.load_state_dict(torch.load("/home/mqr/code/AI-TaxingPolicy/agents/models/bmfac/run105/house_actor.pt"))
        mean_gov_rewards, mean_house_rewards, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years, avg_wealth_stacked, avg_income_stacked = self._evaluate_agent()
        print("mean gov reward:", mean_gov_rewards)


    # do the initial exploration by using the uniform policy
    def _initial_exploration(self, exploration_policy='gaussian'):
        # get the action information of the environment
        global_obs, private_obs = self.envs.reset()
        # past_mean_house_action = self.multiple_households_mean_action()
        for _ in range(self.args.init_exploration_steps):
            if exploration_policy == 'uniform':
                raise NotImplementedError
            elif exploration_policy == 'gaussian':
                with torch.no_grad():
                    gov_action = np.array([0.1/0.5, 0.0/0.05, 0, 0, 0.1]) + np.random.normal(0,0.1, size=(5,))
                    temp = np.zeros((self.args.n_households, 2))
                    temp[:, 0] = 0.9
                    temp[:, 1] = 1 / 3
                    temp += np.random.normal(0,0.1, size=(self.args.n_households,2))

                    hou_action = temp * 2 - 1
                    gov_action = gov_action * 2 - 1
                    past_mean_house_action = self.multiple_households_mean_action(hou_action)[0]

                    action = {self.envs.government.name: gov_action,
                              self.envs.households.name: hou_action}
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)

                # store the episodes
                self.buffer.mf_add(global_obs, private_obs, gov_action, hou_action, past_mean_house_action,
                                gov_reward, house_reward, next_global_obs, next_private_obs, float(done))
                # past_mean_house_action = self.multiple_households_mean_action(hou_action)
                global_obs = next_global_obs
                private_obs = next_private_obs
                if done:
                    # if done, reset the environment
                    global_obs, private_obs = self.envs.reset()
                    # past_mean_house_action = self.multiple_households_mean_action()
        print("Initial exploration has been finished!")

    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
        return obs_tensor

    def _update_network(self, update_agent="households"):
        # smaple batch of samples from the replay buffer
        global_obses, private_obses, gov_actions, hou_actions, past_mean_house_actions, gov_rewards,\
        house_rewards, next_global_obses, next_private_obses, dones = self.buffer.mf_sample(self.args.batch_size)
        # preprocessing the data into the tensors, will support GPU later
        global_obses = torch.tensor(global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        private_obses = torch.tensor(private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_actions = torch.tensor(gov_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        hou_actions = torch.tensor(hou_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        past_mean_house_actions = torch.tensor(past_mean_house_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_rewards = torch.tensor(gov_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        house_rewards = torch.tensor(house_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_global_obses = torch.tensor(next_global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_private_obses = torch.tensor(next_private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - dones, dtype=torch.float32,
                                     device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        # todo update government critic
        next_gov_pi = self.gov_actor(next_global_obses)
        next_gov_action, _ = get_action_info(next_gov_pi, cuda=self.args.cuda).select_actions(reparameterize=True)

        # house_pis = self.house_actor(global_obses, private_obses, gov_actions, past_mean_house_actions, update=True)  # current action
        house_pis = self.house_actor(global_obses, private_obses, gov_actions, update=True)  # current action
        house_actions_info = get_action_info(house_pis, cuda=self.args.cuda)
        house_actions_, house_pre_tanh_value = house_actions_info.select_actions(reparameterize=True)

        # next_hou_pi = self.house_actor(next_global_obses, next_private_obses, next_gov_action, self.get_tensor_mean_action(house_actions_, private_obses), update=True)
        next_hou_pi = self.house_actor(next_global_obses, next_private_obses, next_gov_action, update=True)
        next_hou_action, _ = get_action_info(next_hou_pi, cuda=self.args.cuda).select_actions(reparameterize=True)

        gov_td_target = gov_rewards.reshape(self.args.batch_size,-1) + inverse_dones * self.args.gamma * self.target_gov_qf(next_global_obses, next_gov_action, self.get_tensor_mean_action(next_hou_action, next_private_obses))

        gov_q_value = self.gov_critic(global_obses, gov_actions, past_mean_house_actions)
        gov_td_delta = gov_td_target - gov_q_value
        gov_critic_loss = torch.mean(F.mse_loss(gov_q_value, gov_td_target.detach()))

        n_inverse_dones = inverse_dones.unsqueeze(1).repeat(1, self.args.n_households, 1)
        house_td_target = house_rewards + n_inverse_dones * self.args.gamma * self.target_house_qf(next_global_obses, next_private_obses, next_gov_action, next_hou_action, self.get_tensor_mean_action(next_hou_action, next_private_obses))
        house_q_value = self.house_critic(global_obses, private_obses, gov_actions, hou_actions, past_mean_house_actions)
        house_td_delta = house_td_target - house_q_value
        house_critic_loss = torch.mean(F.mse_loss(house_q_value, house_td_target.detach()))

        # todo government actor
        gov_pis = self.gov_actor(global_obses)
        gov_actions_info = get_action_info(gov_pis, cuda=self.args.cuda)
        gov_actions_, gov_pre_tanh_value = gov_actions_info.select_actions(reparameterize=True)
        gov_log_prob = gov_actions_info.get_log_prob(gov_actions_, gov_pre_tanh_value)
        gov_actor_loss = torch.mean(-gov_log_prob * gov_td_delta.detach())

        # todo households actor
        house_log_prob = house_actions_info.get_log_prob(house_actions_, house_pre_tanh_value)/self.args.n_households
        house_actor_loss = torch.mean(-house_log_prob.sum(2) * house_td_delta.detach().mean(1))


        if update_agent=="households":
            self.house_actor_optim.zero_grad()
            self.house_critc_optim.zero_grad()
            house_actor_loss.backward()
            house_critic_loss.backward()
            self.house_actor_optim.step()
            self.house_critc_optim.step()
        elif update_agent=="government":
            self.gov_actor_optim.zero_grad()
            self.gov_critic_optim.zero_grad()
            gov_actor_loss.backward()
            gov_critic_loss.backward()
            self.gov_actor_optim.step()
            self.gov_critic_optim.step()
        else: # update all
            self.house_actor_optim.zero_grad()
            self.house_critc_optim.zero_grad()
            house_actor_loss.backward()
            house_critic_loss.backward()
            self.house_actor_optim.step()
            self.house_critc_optim.step()

            self.gov_actor_optim.zero_grad()
            self.gov_critic_optim.zero_grad()
            gov_actor_loss.backward()
            gov_critic_loss.backward()
            self.gov_actor_optim.step()
            self.gov_critic_optim.step()

        return gov_actor_loss.item(), gov_critic_loss.item(), house_actor_loss.item(), house_critic_loss.item()

    # update the target network
    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)


    def _evaluate_agent(self):
        total_gov_reward = 0
        total_house_reward = 0
        episode_mean_tax = []
        episode_mean_wealth = []
        episode_mean_post_income = []
        episode_gdp = []
        episode_income_gini = []
        episode_wealth_gini = []
        wealth_stacked_data = []
        income_stacked_data = []
        total_steps = 0

        for _ in range(self.args.eval_episodes):
            global_obs, private_obs = self.eval_env.reset()
            episode_gov_reward = 0
            episode_mean_house_reward = 0
            step_count = 0
            while True:

                with torch.no_grad():
                    action = self._evaluate_get_action(global_obs, private_obs)
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)

                step_count += 1
                episode_gov_reward += gov_reward
                episode_mean_house_reward += np.mean(house_reward)
                episode_mean_tax.append(np.mean(self.eval_env.tax_array))
                episode_mean_wealth.append(np.mean(self.eval_env.households.at_next))
                episode_mean_post_income.append(np.mean(self.eval_env.post_income))
                episode_gdp.append(self.eval_env.per_household_gdp)
                episode_income_gini.append(self.eval_env.income_gini)
                episode_wealth_gini.append(self.eval_env.wealth_gini)
                # wealth_satcked_data:
                wealth_stacked_data.append(self.eval_env.stacked_data(self.eval_env.households.at_next))
                income_stacked_data.append(self.eval_env.stacked_data(self.eval_env.post_income))
                # self.eval_env.render()
                if done:
                    break

                global_obs = next_global_obs
                private_obs = next_private_obs

            total_gov_reward += episode_gov_reward
            total_house_reward += episode_mean_house_reward
            total_steps += step_count
        avg_gov_reward = total_gov_reward / self.args.eval_episodes
        avg_house_reward = total_house_reward / self.args.eval_episodes
        mean_step = total_steps / self.args.eval_episodes
        avg_mean_tax = np.mean(episode_mean_tax)
        avg_mean_wealth = np.mean(episode_mean_wealth)
        avg_mean_post_income = np.mean(episode_mean_post_income)
        avg_gdp = np.mean(episode_gdp)
        avg_income_gini = np.mean(episode_income_gini)
        avg_wealth_gini = np.mean(episode_wealth_gini)
        avg_wealth_stacked = np.mean(wealth_stacked_data, axis=0)
        avg_income_stacked = np.mean(income_stacked_data, axis=0)
        return avg_gov_reward, avg_house_reward, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, mean_step, avg_wealth_stacked, avg_income_stacked

    def _evaluate_get_action(self, global_obs, private_obs):
        global_obs_tensor = self._get_tensor_inputs(global_obs)
        private_obs_tensor = self._get_tensor_inputs(private_obs)
        gov_pi = self.gov_actor(global_obs_tensor)
        gov_action = get_action_info(gov_pi, cuda=self.args.cuda).select_actions(exploration=False,
                                                                                 reparameterize=False)
        hou_pi = self.house_actor(global_obs_tensor, private_obs_tensor, gov_action)
        hou_action = get_action_info(hou_pi, cuda=self.args.cuda).select_actions(exploration=False,
                                                                                 reparameterize=False)

        gov_action = gov_action.detach().cpu().numpy()[0]
        hou_action = hou_action.detach().cpu().numpy()[0]
        if self.fix_gov == True:
            gov_action = np.array([0.1 / 0.2, 0.0 / 0.05, 0, 0, 0.1 / 0.5]) + np.random.normal(0, 0.1, size=(5,))

        action = {self.eval_env.government.name: self.gov_action_max * gov_action,
                  self.eval_env.households.name: self.hou_action_max * hou_action}
        return action
