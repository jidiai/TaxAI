def _evaluate_agent(self):
    np.random.seed(1)
    total_gov_reward = 0
    total_house_reward = 0
    total_steps = 0
    mean_tax = 0
    mean_wealth = 0
    mean_post_income = 0
    gdp = 0
    income_gini = 0
    wealth_gini = 0
    # for epoch_i in range(self.args.eval_episodes):
    for epoch_i in range(1):
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
                # action = self._evaluate_get_action(global_obs, private_obs)
                action = self.test_evaluate_get_action(global_obs, private_obs, step_count)
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
            # if step_count == 1 or step_count == 10 or step_count == 30 or step_count == 300:
            if step_count < 6:
                save_parameters(self.model_path, step_count, epoch_i, self.eval_env)  # 与之前一样 不变
            else:
                break

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



def test_evaluate_get_action(self, global_obs, private_obs, i):
    self.obs = self.obs_concate_numpy(global_obs, private_obs, update=False)
    
    house_values, house_pis = self.households_net(self._get_tensor_inputs(self.obs))
    # select actions
    # gov_action = np.random.random(self.envs.government.action_space.shape[0])
    gov_actions = np.array([[0.99860359, 0.28571885, 0.49025352, 0.59911031, 0.189],
                            [0.89874693, 0.716929, 0.49025352, 0.59911031, 0.189],
                            [0.032421, 0.3282561, 0.49025352, 0.59911031, 0.189],
                            [0.010699, 0.55726823, 0.49025352, 0.59911031, 0.189],
                            [0.76172986, 0.24048432, 0.49025352, 0.59911031, 0.189],
                            [0.84251821, 0.33672109, 0.49025352, 0.59911031, 0.189]])
    gov_action = gov_actions[i]
    print(gov_action)
    
    # ppo
    # house_actions = select_actions(house_pis)
    # input_actions = self.action_wrapper(house_actions)
    # random
    # temp = np.random.random((self.args.n_households, self.envs.households.action_space.shape[1]))
    # input_actions = temp
    # todo GA
    input_actions =

    
    action = {self.envs.government.name: self.gov_action_max * (gov_action * 2 - 1),
              self.envs.households.name: self.hou_action_max * (input_actions * 2 - 1)}
    return action