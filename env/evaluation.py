'''
standard evaluation function
'''

import numpy as np
import torch
import inspect


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


# 获取类的参数
def get_class_parameters(cls):
    parameters = {}
    for param_name, param_value in cls.__dict__.items():
        if not param_name.startswith('__') and not inspect.ismethod(param_value):
            parameters[param_name] = param_value
    return parameters

'''save parameters'''
def save_parameters(path, step, epoch, cls):

    parameters = get_class_parameters(cls)
    file_path = str(path) + '/epoch_' + str(epoch) + 'step_' + str(step) + '_parameters.txt'

    # 将参数保存到文件
    with open(file_path, 'w') as file:
        for param_name, param_value in parameters.items():
            file.write(f"{param_name}: {param_value}\n")

    print(f"参数已保存到文件: {file_path}")