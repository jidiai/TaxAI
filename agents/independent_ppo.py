import numpy as np
import torch
from torch import optim
import os,sys
sys.path.append(os.path.abspath('../..'))
from agents.models import mlp_net
from agents.utils import select_actions, evaluate_actions
from agents.log_path import make_logpath
from datetime import datetime
import os
import copy
import wandb


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class ppo_agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        # start to build the network.
        self.households_net = mlp_net(self.envs.households.observation_space.shape[0], self.envs.households.action_space.shape[0])
        self.households_old_net = copy.deepcopy(self.households_net)
        # if use the cuda...
        if self.args.cuda:
            self.households_net.cuda()
            self.households_old_net.cuda()
        # define the optimizer...
        self.optimizer = optim.Adam(self.households_net.parameters(), self.args.p_lr, eps=self.args.eps)

        # get the observation
        self.batch_ob_shape = (self.args.n_households * self.args.epoch_length, ) + self.envs.households.observation_space.shape
        # self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        # if self.args.env_type == 'mujoco':
        #     self.obs[:] = np.expand_dims(self.running_state(self.envs.reset()), 0)
        # else:
        #     self.obs[:] = self.envs.reset()
        self.dones = np.tile(False, (self.args.n_households, 1))
        # get the action max
        self.gov_action_max = self.envs.government.action_space.high[0]
        self.hou_action_max = self.envs.households.action_space.high[0]

        self.model_path, _ = make_logpath(algo="independent_ppo")
        save_args(path=self.model_path, args=self.args)
        wandb.init(
            config=self.args,
            project="AI_TaxingPolicy",
            entity="ai_tax",
            name=self.model_path.parent.name + "-" + self.model_path.name + '  n=' + str(self.args.n_households),
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

    def obs_concate_numpy(self, global_obs, private_obs, update=True):
        # if update == True:
        #     global_obs = global_obs.unsqueeze(1)
        n_global_obs = np.tile(global_obs, (self.args.n_households, 1))
        return np.concatenate((n_global_obs, private_obs), axis=-1)

    def action_wrapper(self, actions):
        return (actions - np.min(actions, axis=0)) / (np.max(actions, axis=0) - np.min(actions, axis=0))

    # start to train the network...
    def learn(self):
        episode_rewards = np.zeros((self.args.n_households, ), dtype=np.float32)
        global_obs, private_obs = self.envs.reset()
        self.obs = self.obs_concate_numpy(global_obs, private_obs, update=False)
        gov_rew = []
        house_rew = []
        epochs = []

        for update in range(self.args.n_epochs):
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
            self._adjust_learning_rate(update, self.args.n_epochs)
            for step in range(self.args.epoch_length):
                with torch.no_grad():
                    # get tensors
                    values, pis = self.households_net(self._get_tensor_inputs(self.obs))
                # select actions
                house_actions = select_actions(pis)
                input_actions = self.action_wrapper(house_actions)
                gov_action = np.array([0.263, 0.049, 0, 0, 0.189, 0.8])
                # gov_action = np.array([0.263, 0.049, 0.02, 0, 0.189, 0.4])

                action = {self.envs.government.name: self.gov_action_max * (gov_action * 2 - 1),
                          self.envs.households.name: self.hou_action_max * (input_actions*2-1) }

                # start to store information
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(house_actions)
                mb_dones.append(self.dones)
                mb_values.append(values.detach().cpu().numpy().squeeze(0))
                # start to excute the actions in the environment
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)

                self.dones = np.tile(done, (self.args.n_households, 1))
                mb_rewards.append(house_reward)
                # clear the observation
                if done:
                    next_global_obs, next_private_obs = self.envs.reset()

                self.obs = self.obs_concate_numpy(next_global_obs, next_private_obs, update=False)
                # process the rewards part -- display the rewards on the screen
                episode_rewards += house_reward.flatten()
                masks = np.array([0.0 if done_ else 1.0 for done_ in self.dones], dtype=np.float32)

                episode_rewards *= masks
            # process the rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            mb_values = np.asarray(mb_values, dtype=np.float32)

            with torch.no_grad():
                obs_tensor = self._get_tensor_inputs(self.obs)
                last_values, _ = self.households_net(obs_tensor)
                last_values = last_values.detach().cpu().numpy().squeeze(0)
            # start to compute advantages...
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(self.args.epoch_length)):
                if t == self.args.epoch_length - 1:
                    nextnonterminal = 1.0 - np.asarray(self.dones)
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[t + 1]
                    nextvalues = mb_values[t + 1]

                delta = mb_rewards[t] + self.args.ppo_gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.args.ppo_gamma * self.args.ppo_tau * nextnonterminal * lastgaelam
            mb_returns = mb_advs + mb_values
            # after compute the returns, let's process the rollouts
            mb_obs = mb_obs.swapaxes(0, 1).reshape(self.batch_ob_shape)
            mb_actions = mb_actions.swapaxes(0, 1).reshape(-1, self.envs.households.action_space.shape[0])

            mb_returns = mb_returns.swapaxes(0, 1).flatten()
            mb_advs = mb_advs.swapaxes(0, 1).flatten()
            # before update the network, the old network will try to load the weights
            self.households_old_net.load_state_dict(self.households_net.state_dict())
            # start to update the network
            pl, vl, ent = self._update_network(mb_obs, mb_actions, mb_returns, mb_advs)
            if update % self.args.display_interval == 0:
                # start to do the evaluation
                mean_gov_rewards, mean_house_rewards, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years = self._evaluate_agent()
                # store rewards and step
                now_step = (update + 1) * self.args.epoch_length
                gov_rew.append(mean_gov_rewards)
                house_rew.append(mean_house_rewards)
                np.savetxt(str(self.model_path) + "/gov_reward.txt", gov_rew)
                np.savetxt(str(self.model_path) + "/house_reward.txt", house_rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)

                # GDP + mean utility + wealth distribution + income distribution
                wandb.log({"mean households utility": mean_house_rewards,
                           "goverment utility": mean_gov_rewards,
                           "wealth gini": avg_wealth_gini,
                           "income gini": avg_income_gini,
                           "GDP": avg_gdp,
                           "years": years,
                           "tax per households": avg_mean_tax,
                           "post income per households": avg_mean_post_income,
                           "wealth per households": avg_mean_wealth,
                           "households actor loss": pl,
                           "households critic loss": vl,
                           "steps": now_step})
                print('[{}] Update: {} / {}, Frames: {}, Gov_Rewards: {:.3f}, House_Rewards: {:.3f}, years: {:.3f}, PL: {:.3f},'\
                    'VL: {:.3f}, Ent: {:.3f}'.format(datetime.now(), update, self.args.n_epochs, now_step, mean_gov_rewards, mean_house_rewards, years, pl, vl, ent))
                # save the model
                torch.save(self.households_net.state_dict(), str(self.model_path) + '/house_net.pt')
        wandb.finish()


    # update the network
    def _update_network(self, obs, actions, returns, advantages):
        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.args.batch_size
        for _ in range(self.args.update_epoch):
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                mbinds = inds[start:end]
                mb_obs = obs[mbinds]
                mb_actions = actions[mbinds]
                mb_returns = returns[mbinds]
                mb_advs = advantages[mbinds]
                # convert minibatches to tensor
                mb_obs = self._get_tensor_inputs(mb_obs)
                mb_actions = torch.tensor(mb_actions, dtype=torch.float32)
                mb_returns = torch.tensor(mb_returns, dtype=torch.float32).unsqueeze(1)
                mb_advs = torch.tensor(mb_advs, dtype=torch.float32).unsqueeze(1)
                # normalize adv
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                if self.args.cuda:
                    mb_actions = mb_actions.cuda()
                    mb_returns = mb_returns.cuda()
                    mb_advs = mb_advs.cuda()
                # start to get values
                mb_values, pis = self.households_net(mb_obs)
                # start to calculate the value loss...
                value_loss = (mb_returns - mb_values).pow(2).mean()
                # start to calculate the policy loss
                with torch.no_grad():
                    _, old_pis = self.households_old_net(mb_obs)
                    # get the old log probs
                    old_log_prob, _ = evaluate_actions(old_pis, mb_actions)
                    old_log_prob = old_log_prob.detach()
                # evaluate the current policy
                log_prob, ent_loss = evaluate_actions(pis, mb_actions)
                prob_ratio = torch.exp(log_prob - old_log_prob)
                # surr1
                surr1 = prob_ratio * mb_advs
                surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                # final total loss
                total_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef
                # clear the grad buffer
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.households_net.parameters(), self.args.max_grad_norm)
                # update
                self.optimizer.step()
        return policy_loss.item(), value_loss.item(), ent_loss.item()

    # adjust the learning rate
    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.p_lr * lr_frac
        for param_group in self.optimizer.param_groups:
             param_group['lr'] = adjust_lr

    #
    # # evaluate the agent
    # def _evaluate_agent(self):
    #     total_gov_reward = 0
    #     total_house_reward = 0
    #     episode_mean_tax = []
    #     episode_mean_wealth = []
    #     episode_mean_post_income = []
    #     episode_gdp = []
    #     episode_income_gini = []
    #     episode_wealth_gini = []
    #     for _ in range(self.args.eval_episodes):
    #         global_obs, private_obs = self.eval_env.reset()
    #         self.obs = self.obs_concate_numpy(global_obs, private_obs, update=False)
    #         episode_gov_reward = 0
    #         episode_mean_house_reward = 0
    #
    #         while True:
    #             with torch.no_grad():
    #
    #                 # start to excute the actions in the environment
    #                 next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
    #
    #             episode_gov_reward += gov_reward
    #             episode_mean_house_reward += np.mean(house_reward)
    #             episode_mean_tax.append(np.mean(self.eval_env.tax_array))
    #             episode_mean_wealth.append(np.mean(self.eval_env.households.at_next))
    #             episode_mean_post_income.append(np.mean(self.eval_env.post_income))
    #             episode_gdp.append(self.eval_env.per_household_gdp)
    #             episode_income_gini.append(self.eval_env.income_gini)
    #             episode_wealth_gini.append(self.eval_env.wealth_gini)
    #
    #             if done:
    #                 break
    #             global_obs = next_global_obs
    #             private_obs = next_private_obs
    #
    #
    #         total_gov_reward += episode_gov_reward
    #         total_house_reward += episode_mean_house_reward
    #
    #     avg_gov_reward = total_gov_reward / self.args.eval_episodes
    #     avg_house_reward = total_house_reward / self.args.eval_episodes
    #     avg_mean_tax = np.mean(episode_mean_tax)
    #     avg_mean_wealth = np.mean(episode_mean_wealth)
    #     avg_mean_post_income = np.mean(episode_mean_post_income)
    #     avg_gdp = np.mean(episode_gdp)
    #     avg_income_gini = np.mean(episode_income_gini)
    #     avg_wealth_gini = np.mean(episode_wealth_gini)
    #     return avg_gov_reward, avg_house_reward, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini


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
        self.obs = self.obs_concate_numpy(global_obs, private_obs, update=False)
        values, pis = self.households_net(self._get_tensor_inputs(self.obs))
        # select actions
        house_actions = select_actions(pis)
        input_actions = self.action_wrapper(house_actions)
        gov_action = np.array([0.263, 0.049, 0, 0, 0.189, 0.8])

        action = {self.envs.government.name: self.gov_action_max * (gov_action * 2 - 1),
                  self.envs.households.name: self.hou_action_max * (input_actions * 2 - 1)}
        return action

