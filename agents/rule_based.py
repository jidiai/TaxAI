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
        self.args = args

        # get the action max
        self.gov_action_max = self.envs.government.action_space.high[0]
        self.hou_action_max = self.envs.households.action_space.high[0]

        self.model_path, _ = make_logpath(algo="rule_based")
        save_args(path=self.model_path, args=self.args)
        wandb.init(
            config=self.args,
            project="AI_TaxingPolicy",
            entity="ai_tax",
            name=self.model_path.parent.name + "-" +self.model_path.name +'  n='+ str(self.args.n_households),
            dir=str(self.model_path),
            job_type="training",
            reinit=True
        )

    def learn(self):
        # for loop
        global_timesteps = 0
        # reset the environment
        global_obs, private_obs = self.envs.reset()
        gov_rew = []
        house_rew = []
        epochs = []
        for epoch in range(self.args.n_epochs):
            # for each epoch, it will reset the environment
            # for t in range(self.args.epoch_length):
            #     gov_action = np.array([0.263, 0.049, 0, 0, 0])
            #     temp = np.zeros((self.args.n_households,2))
            #     temp[:, 0] = 0.2
            #     temp[:, 1] = 1/3
            #     hou_action = temp
            #
            #     action = {self.envs.government.name: self.gov_action_max * gov_action,
            #               self.envs.households.name: self.hou_action_max * hou_action}
            #     next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)
            #     global_timesteps += 1
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
                           "steps": now_step})
                print(
                    '[{}] Epoch: {} / {}, Frames: {}, gov_Rewards: {:.3f}, house_Rewards: {:.3f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length, mean_gov_rewards, mean_house_rewards))

        wandb.finish()

    # evaluate the agent
    def _evaluate_agent(self):
        total_gov_reward = 0
        total_house_reward = 0
        for _ in range(self.args.eval_episodes):
            episode_gov_reward = 0
            episode_mean_house_reward = 0
            while True:
                with torch.no_grad():
                    gov_action = np.array([0.263, 0.049, 0, 0, 0])
                    temp = np.zeros((self.args.n_households, 2))
                    temp[:, 0] = 0.2
                    temp[:, 1] = 1 / 3
                    hou_action = temp *2-1

                    action = {self.envs.government.name: self.gov_action_max * (gov_action*2-1),
                              self.envs.households.name: self.hou_action_max * hou_action}
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)

                episode_gov_reward += gov_reward
                episode_mean_house_reward += np.mean(house_reward)
                if done:
                    break

            total_gov_reward += episode_gov_reward
            total_house_reward += episode_mean_house_reward
        avg_gov_reward = total_gov_reward / self.args.eval_episodes
        avg_house_reward = total_house_reward / self.args.eval_episodes
        return avg_gov_reward[0], avg_house_reward
