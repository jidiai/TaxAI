import copy
import numpy as np
import torch
import os,sys
import wandb
import time
sys.path.append(os.path.abspath('../..'))

from agents.log_path import make_logpath
from datetime import datetime
from tensorboardX import SummaryWriter
from env.evaluation import save_parameters

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
                project="TaxAI",
                entity="taxai",
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
        
        for epoch in range(10):
        # for epoch in range(self.args.n_epochs):

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
        total_steps = 0
        mean_tax = 0
        mean_wealth = 0
        mean_post_income = 0
        gdp = 0
        income_gini = 0
        wealth_gini = 0
        step_time = 0
        epochs_time = 0
        for epoch_i in range(self.args.eval_episodes):
        # for epoch_i in range(100):
            global_obs, private_obs = self.eval_env.reset()
            episode_gov_reward = 0
            episode_mean_house_reward = 0
            step_count = 0
            episode_mean_tax = []
            episode_mean_wealth = []
            episode_mean_post_income = []
            episode_gdp = []
            episode_income_gini = []
            episode_wealth_gini = []
            episode_time = []
        
            while True:
                with torch.no_grad():
                    action = self._evaluate_get_action(global_obs, private_obs)
                    start_time = time.time()
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    end_time = time.time()
                    execution_time = end_time - start_time
                
                episode_time.append(execution_time)
                step_count += 1
                episode_gov_reward += gov_reward
                episode_mean_house_reward += np.mean(house_reward)
                episode_mean_tax.append(np.mean(self.eval_env.tax_array))
                episode_mean_wealth.append(np.mean(self.eval_env.households.at_next))
                episode_mean_post_income.append(np.mean(self.eval_env.post_income))
                episode_gdp.append(self.eval_env.per_household_gdp)
                episode_income_gini.append(self.eval_env.income_gini)
                episode_wealth_gini.append(self.eval_env.wealth_gini)
                if step_count == 1 or step_count == 100 or step_count == 200 or step_count == 300:
                    save_parameters(self.model_path, step_count, epoch_i, self.eval_env)
                if done:
                    break
            
                global_obs = next_global_obs
                private_obs = next_private_obs
            
            step_time += np.mean(episode_time)
            epochs_time += np.sum(episode_time)
            total_gov_reward += episode_gov_reward
            total_house_reward += episode_mean_house_reward
            total_steps += step_count
            mean_tax += np.mean(episode_mean_tax)
            mean_wealth += np.mean(episode_mean_wealth)
            mean_post_income += np.mean(episode_mean_post_income)
            gdp += np.mean(episode_gdp)
            income_gini += np.mean(episode_income_gini)
            wealth_gini += np.mean(episode_wealth_gini)
    
        avg_gov_reward = total_gov_reward / self.args.eval_episodes
        avg_house_reward = total_house_reward / self.args.eval_episodes
        mean_step = total_steps / self.args.eval_episodes
        avg_mean_tax = mean_tax / self.args.eval_episodes
        avg_mean_wealth = mean_wealth / self.args.eval_episodes
        avg_mean_post_income = mean_post_income / self.args.eval_episodes
        avg_gdp = gdp / self.args.eval_episodes
        avg_income_gini = income_gini / self.args.eval_episodes
        avg_wealth_gini = wealth_gini / self.args.eval_episodes
        step_time /= self.args.eval_episodes
        epochs_time /= self.args.eval_episodes
        print("each step:", 1/step_time)
        print("each epoch:", 1/epochs_time)
        return avg_gov_reward, avg_house_reward, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, \
               avg_wealth_gini, mean_step
    def _evaluate_get_action(self, global_obs, private_obs):
        gov_action = np.array([0.23, 0.01, 0.5, 0.01, 0.189/0.5])
        # gov_action = np.array([0, 0., 0, 0, 0/0.5])
        # gov_action = np.random.random(5)
        
        temp = np.random.random((self.args.n_households, self.envs.households.action_space.shape[1]))
        # hou_action = temp * 2 - 1
        
        # temp[:, 0] = 0.7
        # temp[:, 1] = 1 / 3

        hou_action = temp * 2 - 1

        action = {self.eval_env.government.name: self.gov_action_max * (gov_action * 2 - 1),
                  self.eval_env.households.name: self.hou_action_max * hou_action}
        return action