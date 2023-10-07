'''
standard evaluation function
'''

import numpy as np
import torch
import inspect


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
        episode_gov_reward = 0
        episode_mean_house_reward = 0
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
            if step_count == 1 or step_count == 10 or step_count == 30 or step_count == 300:
                save_parameters(self.model_path, step_count, epoch_i, self.eval_env)
            
            global_obs = next_global_obs
            private_obs = next_private_obs
        
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
    return avg_gov_reward, avg_house_reward, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, \
           avg_wealth_gini, mean_step

def get_class_parameters(cls):
    parameters = {}
    for param_name, param_value in cls.__dict__.items():
        if not param_name.startswith('__') and not inspect.ismethod(param_value):
            parameters[param_name] = param_value
    return parameters

'''save parameters'''
def save_parameters(path, step, epoch, cls):

    parameters = get_class_parameters(cls)
    households_para = get_class_parameters(cls.households)
    file_path = str(path) + '/epoch_' + str(epoch) + '_step_' + str(step) + '_' + str(households_para['n_households']) +'_' + parameters["gov_task"] +'_parameters.pkl'

    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump(parameters, f)
