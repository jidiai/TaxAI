import numpy as np
import torch
from torch import optim
from agents.ppo.models import mlp_net
from agents.ppo.utils import select_actions, evaluate_actions
from datetime import datetime
from agents.log_path import make_logpath
from env.evaluation import save_parameters
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

        self.net = mlp_net(self.envs.households.observation_space.shape[0], self.envs.households.action_space.shape[1])
        self.old_net = copy.deepcopy(self.net)
        # if use the cuda...
        if self.args.cuda:
            self.net.cuda()
            self.old_net.cuda()
        # define the optimizer...
        self.optimizer = optim.Adam(self.net.parameters(), self.args.p_lr, eps=self.args.eps)
        self.dones = np.tile(False, 1)
        self.gov_action_max = self.envs.government.action_space.high[0]
        self.hou_action_max = self.envs.households.action_space.high[0]
        self.optim_agent = 3
        
        
        self.model_path, _ = make_logpath(algo="ppo",n=self.args.n_households)
        save_args(path=self.model_path, args=self.args)
        self.wandb = False
        if self.wandb == True:
            wandb.init(
                config=self.args,
                project="TaxAI",  # please update as your project
                entity="taxai",
                name=self.model_path.parent.parent.name + "-" + self.model_path.name + '  n=' + str(self.args.n_households),
                dir=str(self.model_path),
                job_type="training",
                reinit=True
                )

    
    def observation_wrapper(self, global_obs, private_obs):
        # global
        global_obs[0] /= 1e7
        global_obs[1] /= 1e5
        global_obs[3] /= 1e5
        global_obs[4] /= 1e5
        private_obs[:, 1] /= 1e5
        return global_obs, private_obs
    

    def obs_concate_numpy(self, global_obs, private_obs, update=True):
        n_global_obs = np.tile(global_obs, (self.args.n_households, 1))
        return np.concatenate((n_global_obs, private_obs), axis=-1)
    
    def house_action(self, action, i):
        random_action = np.random.random((self.args.n_households, self.envs.households.action_space.shape[1]))
        random_action[i] = action
        return random_action
    
    # start to train the network...
    def learn(self):
        global_obs, private_obs = self.envs.reset()
        global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
        self.obs = self.obs_concate_numpy(global_obs, private_obs, update=False)
        gov_rew = []
        house_rew = []
        epochs = []

        episode_rewards = np.zeros((1, ), dtype=np.float32)
        final_rewards = np.zeros((1,), dtype=np.float32)
        for update in range(self.args.n_epochs):
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
            self._adjust_learning_rate(update, self.args.n_epochs)
            for step in range(self.args.epoch_length):
                with torch.no_grad():
                    # get tensors
                    house_values, house_pis = self.net(self._get_tensor_inputs(self.obs[self.optim_agent]))
                # select actions
                actions = select_actions(house_pis)
                input_actions = actions.copy()
                input_actions = self.house_action(input_actions, self.optim_agent)
                gov_actions = np.random.random(self.envs.government.action_space.shape[0])
                action = {self.envs.government.name: self.gov_action_max * (gov_actions * 2 - 1),
                          self.envs.households.name: self.hou_action_max * (input_actions * 2 - 1)}
                
                # start to store information
                mb_obs.append(np.copy(self.obs[self.optim_agent]))
                mb_actions.append(actions)
                mb_dones.append(self.dones)
                mb_values.append(house_values.detach().cpu().numpy().squeeze())
                # start to excute the actions in the environment
                next_global_obs, next_private_obs, gov_reward, house_reward, dones = self.envs.step(action)
                next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)
               
                dones = np.array([dones])
                rewards = np.array([house_reward[self.optim_agent]])
                self.dones = dones
                mb_rewards.append(rewards)
                # clear the observation
                for n, done in enumerate(dones):
                    if done:
                        next_global_obs, next_private_obs = self.envs.reset()
                        next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)
                self.obs = self.obs_concate_numpy(next_global_obs, next_private_obs, update=False)
                # process the rewards part -- display the rewards on the screen
                episode_rewards += house_reward[self.optim_agent]
                masks = np.array([0.0 if done_ else 1.0 for done_ in dones], dtype=np.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks
            # process the rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            mb_values = np.expand_dims(mb_values, 1)
            # compute the last state value
            with torch.no_grad():
        
                last_values, _ = self.net(self._get_tensor_inputs(self.obs[self.optim_agent]))
                last_values = last_values.detach().cpu().numpy().squeeze()
            # start to compute advantages...
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(self.args.epoch_length)):
                if t == self.args.epoch_length - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[t + 1]
                    nextvalues = mb_values[t + 1]
                delta = mb_rewards[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam
            mb_returns = mb_advs + mb_values
            mb_returns = mb_returns.swapaxes(0, 1).flatten()
            mb_advs = mb_advs.swapaxes(0, 1).flatten()
            # before update the network, the old network will try to load the weights
            self.old_net.load_state_dict(self.net.state_dict())
            # start to update the network
            pl, vl, ent = self._update_network(mb_obs, mb_actions, mb_returns, mb_advs)
            # display the training information
            if update % self.args.display_interval == 0:
                # start to do the evaluation
                mean_gov_rewards, each_house_utility, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years = self._evaluate_agent()
                # store rewards and step
                mean_house_rewards = np.sum(each_house_utility)
                now_step = (update + 1) * self.args.epoch_length
                gov_rew.append(mean_gov_rewards)
                house_rew.append(mean_house_rewards)
                np.savetxt(str(self.model_path) + "/gov_reward.txt", gov_rew)
                np.savetxt(str(self.model_path) + "/house_reward.txt", house_rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)

                # GDP + mean utility + wealth distribution + income distribution
                if self.wandb == True:
                    wandb.log({"mean households utility": mean_house_rewards,
                           "goverment utility": mean_gov_rewards,
                           "house1" : each_house_utility[0],
                           "house2" : each_house_utility[1],
                           "house3" : each_house_utility[2],
                           "house4" : each_house_utility[3],
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
                torch.save(self.net.state_dict(), str(self.model_path) + '/house_net.pt')
        if self.wandb == True:
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
                mb_values, pis = self.net(mb_obs)
                # start to calculate the value loss...
                value_loss = (mb_returns - mb_values).pow(2).mean()
                # start to calculate the policy loss
                with torch.no_grad():
                    _, old_pis = self.old_net(mb_obs)
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
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.max_grad_norm)
                # update
                self.optimizer.step()
        return policy_loss.item(), value_loss.item(), ent_loss.item()

    # convert the numpy array to tensors
    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
        return obs_tensor

    # adjust the learning rate
    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.p_lr * lr_frac
        for param_group in self.optimizer.param_groups:
             param_group['lr'] = adjust_lr

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
        for epoch_i in range(self.args.eval_episodes):
            global_obs, private_obs = self.eval_env.reset()
            global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
        
            episode_gov_reward = 0
            episode_mean_house_reward = []
            step_count = 0
            episode_mean_tax = []
            episode_mean_wealth = []
            episode_mean_post_income = []
            episode_gdp = []
            episode_income_gini = []
            episode_wealth_gini = []
        
            while True:
                with torch.no_grad():
                    action = self._evaluate_get_action(global_obs, private_obs)
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)
            
                step_count += 1
                episode_gov_reward += gov_reward
                episode_mean_house_reward.append(house_reward)
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
        
            total_gov_reward += episode_gov_reward
            total_house_reward += np.sum(episode_mean_house_reward, axis=0)
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
        return avg_gov_reward, avg_house_reward, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, \
               avg_wealth_gini, mean_step

    def _evaluate_get_action(self, global_obs, private_obs):
        self.obs = self.obs_concate_numpy(global_obs, private_obs, update=False)
        house_values, house_pis = self.net(self._get_tensor_inputs(self.obs[self.optim_agent]))
        # select actions
        actions = select_actions(house_pis)
        input_actions = actions.copy()
        input_actions = self.house_action(input_actions, self.optim_agent)
        gov_actions = np.random.random(self.envs.government.action_space.shape[0])
        action = {self.envs.government.name: self.gov_action_max * (gov_actions * 2 - 1),
                  self.envs.households.name: self.hou_action_max * (input_actions * 2 - 1)}
        return action


