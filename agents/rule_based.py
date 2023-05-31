import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import os,sys
import wandb
sys.path.append(os.path.abspath('../..'))

from agents.models import SharedAgent, SharedCritic, Actor, Critic
from agents.log_path import make_logpath
from utils.experience_replay import replay_buffer
from agents.utils import get_action_info
from datetime import datetime
from tensorboardX import SummaryWriter
from env.evaluation import save_parameters
# os.environ["SDL_VIDEODRIVER"] = "directfb"
import pygame
torch.autograd.set_detect_anomaly(True)

def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class rule_agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args

        # get the action max
        self.gov_action_max = self.envs.government.action_space.high[0]
        self.hou_action_max = self.envs.households.action_space.high[0]

        self.model_path, _ = make_logpath(algo="rule_based",n=self.args.n_households)
        save_args(path=self.model_path, args=self.args)
        self.wandb = False
        if self.wandb:
            wandb.init(
                config=self.args,
                project="AI_TaxingPolicy",
                entity="ai_tax",
                name=self.model_path.parent.parent.name + "-" +self.model_path.name +'  n='+ str(self.args.n_households),
                dir=str(self.model_path),
                job_type="training",
                reinit=True
            )
        # pygame.init()

    def learn(self):

        gov_rew = []
        house_rew = []
        epochs = []
        wealth_stack = []
        income_stack = []
        
        for epoch in range(1000):


            if epoch % self.args.display_interval == 0:

                # start to do the evaluation
                mean_gov_rewards, mean_house_rewards, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years, avg_wealth_stacked, avg_income_stacked, avg_wealth = self._evaluate_agent()
                # store rewards and step
                now_step = (epoch + 1) * self.args.epoch_length
                gov_rew.append(mean_gov_rewards)
                house_rew.append(mean_house_rewards)
                np.savetxt(str(self.model_path) + "/gov_reward.txt", gov_rew)
                np.savetxt(str(self.model_path) + "/house_reward.txt", house_rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)
                wealth_stack.append(avg_wealth_stacked)
                income_stack.append(avg_income_stacked)

                np.savetxt(str(self.model_path) + "/wealth_stack.txt", wealth_stack)
                np.savetxt(str(self.model_path) + "/income_stack.txt", income_stack)
                # np.savetxt(str(self.model_path) + "/wealth_set.txt", avg_wealth.reshape(5,100))

                if self.wandb:
                    # GDP + mean utility + wealth distribution + income distribution
                    wandb.log({"mean households utility": mean_house_rewards,
                               "goverment utility": mean_gov_rewards,
                               "years": years,
                               "wealth gini": avg_wealth_gini,
                               "income gini": avg_income_gini,
                               "GDP": avg_gdp,
                               "tax per households": avg_mean_tax,
                               "post income per households": avg_mean_post_income,
                               "wealth per households": avg_mean_wealth,
                               "steps": now_step})
                print(
                    '[{}] Epoch: {} / {}, Frames: {}, gov_Rewards: {:.3f}, house_Rewards: {:.3f}, years: {:.3f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length, mean_gov_rewards, mean_house_rewards, years))
        if self.wandb:
            wandb.finish()

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
        wealth_sum = []
        
        total_steps = 0

        for _ in range(self.args.eval_episodes):
            global_obs, private_obs = self.eval_env.reset()
            episode_gov_reward = 0
            episode_mean_house_reward = 0
            step_count = 0
            welath_set = []
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
                # if step_count == 1 or step_count == 5 or step_count == 10 or step_count == 15 or step_count == 20:
                #     welath_set.append(self.eval_env.households.at_next)
                # self.eval_env.render()
                if done:
                    break

                global_obs = next_global_obs
                private_obs = next_private_obs

            total_gov_reward += episode_gov_reward
            total_house_reward += episode_mean_house_reward
            total_steps += step_count
            # wealth_sum.append(welath_set)

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
        # avg_wealth_set = np.mean(wealth_sum, axis=0)
        avg_wealth_set = 0
        return avg_gov_reward, avg_house_reward, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, mean_step, avg_wealth_stacked, avg_income_stacked, avg_wealth_set

    def _evaluate_get_action(self, global_obs, private_obs):
        # gov_action = np.array([0.5, 0.01, 0.5, 0.01, 0.189/0.5])
        gov_action = np.array([0.5, 0., 0.5, 0, 0/0.5])
        temp = np.random.random((self.args.n_households, self.envs.households.action_space.shape[1]))

        hou_action = temp * 2 - 1
        
        action = {self.eval_env.government.name: self.gov_action_max * (gov_action * 2 - 1),
                  self.eval_env.households.name: self.hou_action_max * hou_action}
        return action