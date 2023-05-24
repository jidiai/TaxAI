import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import os,sys
import wandb
from .maddpg import MADDPG
sys.path.append(os.path.abspath('../..'))
from agents.log_path import make_logpath
from utils.experience_replay import replay_buffer
from agents.utils import get_action_info
from datetime import datetime

torch.autograd.set_detect_anomaly(True)

def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

'''
maddpg
'''
class maddpg_agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args
        self.eval_env = copy.copy(envs)
        # todo add
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.args.n_agents = self.envs.households.n_households + 1

        # # start to build the network.
        self.args.gov_obs_dim = self.envs.government.observation_space.shape[0]
        self.args.gov_action_dim = self.envs.government.action_space.shape[0]
        self.args.house_obs_dim = self.envs.households.observation_space.shape[0]
        self.args.house_action_dim = self.envs.households.action_space.shape[1]
        self.agents = self._init_agents()

        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)

        self.model_path, _ = make_logpath(algo="maddpg")
        save_args(path=self.model_path, args=self.args)
        self.fix_gov = True
        self.wandb = True
        if self.wandb:
            wandb.init(
                config=self.args,
                project="AI_TaxingPolicy",
                entity="ai_tax",
                name=self.model_path.parent.name + "-"+ self.model_path.name +'  n='+ str(self.args.n_households),
                dir=str(self.model_path),
                job_type="training",
                reinit=True
            )
    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = MADDPG(self.args, i)
            agents.append(agent)
        return agents

    def learn(self):
        # reset the environment
        global_obs, private_obs = self.envs.reset()
        gov_rew = []
        house_rew = []
        epochs = []
        agent_list = ["households", "government"]
        update_index = 0

        for epoch in range(self.args.n_epochs):
            print("epoch:", epoch)
            # for each epoch, it will reset the environment
            for t in range(self.args.epoch_length):
                '''
                for each agent, get its action from observation
                '''
                global_obs_tensor = self._get_tensor_inputs(global_obs)
                private_obs_tensor = self._get_tensor_inputs(private_obs)
                hou_action = np.zeros((self.envs.households.n_households, self.args.house_action_dim))
                for agent_id, agent in enumerate(self.agents):
                    if agent_id == self.args.n_agents-1:  # government agent
                        gov_action = agent.select_action(global_obs_tensor, self.noise, self.epsilon)
                    else:   # households agent
                        obs = torch.cat([global_obs_tensor, private_obs_tensor[agent_id]], dim=-1)
                        hou_action[agent_id] = agent.select_action(obs, self.noise, self.epsilon)

                action = {self.envs.government.name: gov_action,
                          self.envs.households.name: hou_action}
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)

                # store the episodes
                self.buffer.add(global_obs, private_obs, gov_action, hou_action, gov_reward, house_reward,
                                next_global_obs, next_private_obs, float(done))
                # past_mean_house_action = self.multiple_households_mean_action(hou_action)
                global_obs = next_global_obs
                private_obs = next_private_obs
                if done:
                    # if done, reset the environment
                    global_obs, private_obs = self.envs.reset()

            # after collect the samples, start to update the network
            transitions = self.buffer.sample(self.args.batch_size)
            sum_actor_loss = 0
            sum_critic_loss = 0
            for agent in self.agents:
                other_agents = self.agents.copy()
                other_agents.remove(agent)
                actor_loss, critic_loss = agent.train(transitions, other_agents)
                sum_actor_loss += actor_loss
                sum_critic_loss += critic_loss


            # print the log information
            if epoch % self.args.display_interval == 0:
                # start to do the evaluation
                mean_gov_rewards, mean_house_rewards, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years = self._evaluate_agent()
                # store rewards and step
                now_step = (epoch + 1) * self.args.epoch_length
                gov_rew.append(mean_gov_rewards)
                house_rew.append(mean_house_rewards)
                np.savetxt(str(self.model_path) + "/gov_reward.txt", gov_rew)
                np.savetxt(str(self.model_path) + "/house_reward.txt", house_rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)
                if self.wandb:
                    wandb.log({"mean households utility": mean_house_rewards,
                               "goverment utility": mean_gov_rewards,
                               "years": years,
                               "wealth gini": avg_wealth_gini,
                               "income gini": avg_income_gini,
                               "GDP": avg_gdp,
                               "tax per households": avg_mean_tax,
                               "post income per households": avg_mean_post_income,
                               "wealth per households": avg_mean_wealth,
                               "actor loss": sum_actor_loss,
                               "critic loss": sum_actor_loss,
                               "steps": now_step})

                print(
                    '[{}] Epoch: {} / {}, Frames: {}, gov_Rewards: {:.3f}, house_Rewards: {:.3f}, years:{:.3f}, actor_loss: {:.3f}, critic_loss: {:.3f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length, mean_gov_rewards, mean_house_rewards,years, sum_actor_loss, sum_critic_loss))
                # save models
            if epoch % self.args.save_interval == 0:
                for agent_i in range(len(self.agents)):
                    torch.save(self.agents[agent_i].actor_network.state_dict(), str(self.model_path) + '/agent_'+str(agent_i)+'.pt')
                # torch.save(self.house_actor.state_dict(), str(self.model_path) + '/house_actor.pt')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
        if self.wandb:
            wandb.finish()

    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        return obs_tensor

    def _evaluate_agent(self):
        total_gov_reward = 0
        total_house_reward = 0
        episode_mean_tax = []
        episode_mean_wealth = []
        episode_mean_post_income = []
        episode_gdp = []
        episode_income_gini = []
        episode_wealth_gini = []
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
        return avg_gov_reward, avg_house_reward, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, mean_step

    def _evaluate_get_action(self, global_obs, private_obs):
        global_obs_tensor = self._get_tensor_inputs(global_obs)
        private_obs_tensor = self._get_tensor_inputs(private_obs)
        hou_action = np.zeros((self.envs.households.n_households, self.args.house_action_dim))
        for agent_id, agent in enumerate(self.agents):
            if agent_id == self.args.n_agents - 1:  # government agent
                gov_action = agent.select_action(global_obs_tensor, self.noise, self.epsilon)
            else:  # households agent
                obs = torch.cat([global_obs_tensor, private_obs_tensor[agent_id]], dim=-1)
                hou_action[agent_id] = agent.select_action(obs, self.noise, self.epsilon)
    
        action = {self.envs.government.name: gov_action,
                  self.envs.households.name: hou_action}
        return action
