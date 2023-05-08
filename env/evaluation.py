'''
standard evaluation function
'''

import numpy as np
import torch

def _evaluate_agent(env, eval_episodes):
    total_gov_reward = 0
    total_house_reward = 0
    episode_mean_tax = []
    episode_mean_wealth = []
    episode_mean_post_income = []
    episode_gdp = []
    episode_income_gini = []
    episode_wealth_gini = []
    total_steps = 0

    for _ in range(eval_episodes):
        global_obs, private_obs = env.reset()
        episode_gov_reward = 0
        episode_mean_house_reward = 0
        step_count = 0
        while True:

            with torch.no_grad():
                action = _evaluate_get_action(global_obs, private_obs)
                next_global_obs, next_private_obs, gov_reward, house_reward, done = env.step(action)

            step_count += 1
            episode_gov_reward += gov_reward
            episode_mean_house_reward += np.mean(house_reward)
            episode_mean_tax.append(np.mean(env.tax_array))
            episode_mean_wealth.append(np.mean(env.households.at_next))
            episode_mean_post_income.append(np.mean(env.post_income))
            episode_gdp.append(env.per_household_gdp)
            episode_income_gini.append(env.income_gini)
            episode_wealth_gini.append(env.wealth_gini)

            if done:
                break
            global_obs = next_global_obs
            private_obs = next_private_obs

        total_gov_reward += episode_gov_reward
        total_house_reward += episode_mean_house_reward
        total_steps += step_count

    avg_gov_reward = total_gov_reward / eval_episodes
    avg_house_reward = total_house_reward / eval_episodes
    mean_step = total_steps / eval_episodes
    avg_mean_tax = np.mean(episode_mean_tax)
    avg_mean_wealth = np.mean(episode_mean_wealth)
    avg_mean_post_income = np.mean(episode_mean_post_income)
    avg_gdp = np.mean(episode_gdp)
    avg_income_gini = np.mean(episode_income_gini)
    avg_wealth_gini = np.mean(episode_wealth_gini)
    return avg_gov_reward, avg_house_reward, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, mean_step


def _evaluate_get_action(global_obs, private_obs):
    pass
    # todo: get gov_action, hou_action
    # gov_action =
    # hou_action =

    # action = {self.eval_env.government.name: self.gov_action_max * (gov_action * 2 - 1),
    #           self.eval_env.households.name: self.hou_action_max * hou_action}
    # return action