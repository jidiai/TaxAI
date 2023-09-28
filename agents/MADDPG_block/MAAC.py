import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import csv
import os,sys
import wandb
from .maddpg import MADDPG
sys.path.append(os.path.abspath('../..'))
from agents.log_path import make_logpath
from utils.experience_replay import replay_buffer
from agents.utils import get_action_info
from datetime import datetime
from env.evaluation import save_parameters

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
        self.args.agent_block_num = 4
        self.agents = self._init_agents()

        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)

        self.model_path, _ = make_logpath(algo="maddpg",n=self.args.n_households)
        save_args(path=self.model_path, args=self.args)
        self.fix_gov = True
        # todo add new names
        self.indicators_name = ["gov_reward", "mean_utility", "years", "total_income", "income_10", "income_50",
                                "income_100", "total_tax", "income_tax", "income_tax_10", "income_tax_50",
                                "income_tax_100", "total_wealth", "wealth_10", "wealth_50", "wealth_100", "wealth_tax",
                                "wealth_tax_10", "wealth_tax_50", "wealth_tax_100",
                                "per_gdp", "income_gini", "wealth_gini", "wage", "total_labor", "labor_10", "labor_50",
                                "labor_100", "sw_10", "sw_50", "sw_100",
                                "total_consumption", "consumption_10", "consumption_50", "consumption_100", "Bt", "Kt","Gt_prob", "income_tau", "income_xi", "wealth_tau", "wealth_xi"]
        
        self.wandb = True
        if self.wandb:
            wandb.init(
                config=self.args,
                project="AI_TaxingPolicy",
                entity="ai_tax",
                name=self.model_path.parent.parent.name + "-"+ self.model_path.name +'  n='+ str(self.args.n_households),
                dir=str(self.model_path),
                job_type="training",
                reinit=True
            )
    def _init_agents(self):
        agents = []
        for i in range(self.args.agent_block_num):  # 3 households group + 1 government
            agent = MADDPG(self.args, i)
            agents.append(agent)
        return agents
    
    def observation_wrapper(self, global_obs, private_obs):
        # global
        global_obs[0] /= 1e7
        global_obs[1] /= 1e5
        global_obs[3] /= 1e5
        global_obs[4] /= 1e5
        private_obs[:, 1] /= 1e5
        return global_obs, private_obs

    def learn(self):
        # reset the environment
        global_obs, private_obs = self.envs.reset()
        global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
        gov_rew = []
        house_rew = []
        epochs = []
        sum_actor_loss = 0
        sum_critic_loss = 0

        for epoch in range(self.args.n_epochs):
        # for epoch in range(1):
            # for each epoch, it will reset the environment
            for t in range(self.args.epoch_length):
                '''
                for each agent, get its action from observation
                '''
                global_obs_tensor = self._get_tensor_inputs(global_obs)
                private_obs_tensor = self._get_tensor_inputs(private_obs)
                hou_action = np.zeros((self.envs.households.n_households, self.args.house_action_dim))
                gov_action = self.agents[-1].select_action(global_obs_tensor, self.noise, self.epsilon)
                
                n_global_obs = global_obs_tensor.repeat(self.envs.households.n_households, 1)
                obs = torch.cat([n_global_obs, private_obs_tensor], dim=-1)  # torch.cat([global_obs_tensor.repeat(self.envs.households.n_households, 1), private_obs_tensor], dim=-1)
                # 根据wealth 排序observation
                sorted_indices = torch.argsort(obs[:, -1], descending=True)
                sorted_obs = obs[sorted_indices]
                num_set = range(0,self.envs.households.n_households)
                for i in range(self.args.agent_block_num-1):
                    if i ==0:
                        num = num_set[:int(0.1*self.envs.households.n_households)]
                    elif i == 1:
                        num = num_set[int(0.1*self.envs.households.n_households):int(0.5*self.envs.households.n_households)]
                    else:
                        num = num_set[int(0.5 * self.envs.households.n_households):]
                    hou_action[num] = self.agents[i].select_action(sorted_obs[num], self.noise, self.epsilon)

                house_sort_index = sorted_indices.cpu().numpy()
                hou_action = hou_action[np.argsort(house_sort_index)]
                action = {self.envs.government.name: gov_action,
                          self.envs.households.name: hou_action}
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)
                
                next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)
                # todo reward shaping
                # if gov_reward < 0:
                #     gov_reward = np.exp(gov_reward)
                # store the episodes
                self.buffer.add(global_obs, private_obs, gov_action, hou_action, gov_reward, house_reward,
                                next_global_obs, next_private_obs, float(done))
                # past_mean_house_action = self.multiple_households_mean_action(hou_action)
                global_obs = next_global_obs
                private_obs = next_private_obs
                if done:
                    # if done, reset the environment
                    global_obs, private_obs = self.envs.reset()
                    global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)

            # for _ in range(self.args.update_cycles):
                if t % 10 == 0:
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
                economic_idicators_dict = self._evaluate_agent()
                # self.light_episode_evolution()
                # store rewards and step
                now_step = (epoch + 1) * self.args.epoch_length
                gov_rew.append(economic_idicators_dict["gov_rew"])
                # house_rew.append(economic_idicators_dict["mean_utility"])
                house_rew.append(economic_idicators_dict["social_welfare"])
                np.savetxt(str(self.model_path) + "/gov_reward.txt", gov_rew)
                np.savetxt(str(self.model_path) + "/house_reward.txt", house_rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)
             
                if self.wandb:
                    wandb.log(economic_idicators_dict)
                print(
                    '[{}] Epoch: {} / {}, Frames: {}, gov_Rewards: {:.3f}, house_Rewards: {:.3f}, years:{:.3f}, actor_loss: {:.3f}, critic_loss: {:.3f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length, economic_idicators_dict["gov_rew"], economic_idicators_dict["social_welfare"],economic_idicators_dict["years"], sum_actor_loss, sum_critic_loss))
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
    
    def test(self):
        # path = "/home/mqr/code/AI-TaxingPolicy/agents/models/maddpg/100/run17/"
        path = "/home/mqr/code/AI-TaxingPolicy/agents/models/maddpg/100/run78/"
    
        for agent_i in range(len(self.agents)):
            self.agents[agent_i].actor_network.load_state_dict(torch.load(path+ '/agent_' + str(agent_i) + '.pt'))
        # total_gov_reward,total_house_reward,total_steps,tax,income_gini,wealth_gini,gdp,income,income_tax,wealth,wealth_tax,labor,consumption,wage = self.episode_evolution()
 
        self.light_episode_evolution()
  
    
    def _evaluate_agent(self):
        # indicators_num = len(self.indicators_name)
        economic_indicators = []

        for epoch_i in range(self.args.eval_episodes):
            global_obs, private_obs = self.eval_env.reset()
            global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
            episode_indicators = []

            while True:
                with torch.no_grad():
                    action, sort_index = self._evaluate_get_action(global_obs, private_obs)
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                steps = 1
                # 对 10%， 10～50%， 50%～100% 人群分组 找到三组的平均数据
                total_income = np.mean(self.eval_env.post_income)
                income_10 = np.mean(self.eval_env.post_income[sort_index[:10]])
                income_50 = np.mean(self.eval_env.post_income[sort_index[10:50]])
                income_100 = np.mean(self.eval_env.post_income[sort_index[50:]])

                total_tax = np.mean(self.eval_env.tax_array)
                income_tax = np.mean(self.eval_env.income_tax)
                income_tax_10 = np.mean(self.eval_env.income_tax[sort_index[:10]])
                income_tax_50 = np.mean(self.eval_env.income_tax[sort_index[10:50]])
                income_tax_100 = np.mean(self.eval_env.income_tax[sort_index[50:]])

                total_wealth = np.mean(self.eval_env.households.at_next)
                wealth_10 = np.mean(self.eval_env.households.at_next[sort_index[:10]])
                wealth_50 = np.mean(self.eval_env.households.at_next[sort_index[10:50]])
                wealth_100 = np.mean(self.eval_env.households.at_next[sort_index[50:]])

                wealth_tax = np.mean(self.eval_env.asset_tax)
                wealth_tax_10 = np.mean(self.eval_env.asset_tax[sort_index[:10]])
                wealth_tax_50 = np.mean(self.eval_env.asset_tax[sort_index[10:50]])
                wealth_tax_100 = np.mean(self.eval_env.asset_tax[sort_index[50:]])

                per_gdp = self.eval_env.per_household_gdp
                income_gini = self.eval_env.income_gini
                wealth_gini = self.eval_env.wealth_gini
                wage = self.eval_env.WageRate

                total_labor = self.eval_env.Lt
                labor = self.eval_env.households.e * self.eval_env.ht
                labor_10 = np.mean(labor[sort_index[:10]])
                labor_50 = np.mean(labor[sort_index[10:50]])
                labor_100 = np.mean(labor[sort_index[50:]])

                sw = np.mean(house_reward)
                sw_10 = np.mean(house_reward[sort_index[:10]])
                sw_50 = np.mean(house_reward[sort_index[10:50]])
                sw_100 = np.mean(house_reward[sort_index[50:]])

                total_consumption = np.mean(self.eval_env.consumption)
                consumption_10 = np.mean(self.eval_env.consumption[sort_index[:10]])
                consumption_50 = np.mean(self.eval_env.consumption[sort_index[10:50]])
                consumption_100 = np.mean(self.eval_env.consumption[sort_index[50:]])

                Bt = self.eval_env.Bt
                Kt = self.eval_env.Kt
                Gt_prob = self.eval_env.Gt_prob
                income_tau = self.eval_env.government.tau
                income_xi = self.eval_env.government.xi
                wealth_tau = self.eval_env.government.tau_a
                wealth_xi = self.eval_env.government.xi_a


                # todo add new indicators
                # datas = [gov_reward, sw, steps, total_income, income_10, income_50, income_100, total_tax, income_tax, income_tax_10, income_tax_50, income_tax_100, total_wealth,
                #          wealth_10, wealth_50, wealth_100, wealth_tax, wealth_tax_10, wealth_tax_50, wealth_tax_100, per_gdp, income_gini, wealth_gini, wage, total_labor, labor_10,
                #          labor_50, labor_100, sw_10, sw_50, sw_100, total_consumption, consumption_10, consumption_50, consumption_100, Bt, Kt,Gt_prob, income_tau, income_xi, wealth_tau, wealth_xi]
                datas = [per_gdp, sw, steps, gov_reward, income_gini, wealth_gini]
                episode_indicators.append(datas)
                if done:
                    break

                global_obs = next_global_obs
                private_obs = next_private_obs
            years = len(episode_indicators)
            avg_episode_data = np.mean(episode_indicators, axis=0)
            avg_episode_data[1:4] *= years  # gdp 是每一步的 mean
            economic_indicators.append(avg_episode_data)

        avg_indicators = np.mean(economic_indicators, axis=0)
        log_name = ["per_gdp", "social_welfare", "years", "gov_rew", "income_gini", "wealth_gini"]

        return {k: v for k, v in zip(log_name, avg_indicators)}
    
    
    def episode_evolution(self):
        # todo add new names
        self.indicators_name = ["gov_reward", "mean_utility", "years", "total_income", "income_10", "income_50",
                                "income_100", "total_tax", "income_tax", "income_tax_10", "income_tax_50",
                                "income_tax_100", "total_wealth", "wealth_10", "wealth_50", "wealth_100", "wealth_tax",
                                "wealth_tax_10", "wealth_tax_50", "wealth_tax_100",
                                "per_gdp", "income_gini", "wealth_gini", "wage", "total_labor", "labor_10", "labor_50",
                                "labor_100", "sw_10", "sw_50", "sw_100",
                                "total_consumption", "consumption_10", "consumption_50", "consumption_100"]
        # indicators_num = len(self.indicators_name)
        economic_indicators = []
    
        for epoch_i in range(self.args.eval_episodes):
            step_count = 0
            global_obs, private_obs = self.eval_env.reset()
            global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
            episode_indicators = []
        
            while True:
                with torch.no_grad():
                    action, sort_index = self._evaluate_get_action(global_obs, private_obs)
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                step_count += 1
                steps = 1
                # 对 10%， 10～50%， 50%～100% 人群分组 找到三组的平均数据
                total_income = np.mean(self.eval_env.post_income)
                income_10 = np.mean(self.eval_env.post_income[sort_index[:10]])
                income_50 = np.mean(self.eval_env.post_income[sort_index[10:50]])
                income_100 = np.mean(self.eval_env.post_income[sort_index[50:]])
            
                total_tax = np.mean(self.eval_env.tax_array)
                income_tax = np.mean(self.eval_env.income_tax)
                income_tax_10 = np.mean(self.eval_env.income_tax[sort_index[:10]])
                income_tax_50 = np.mean(self.eval_env.income_tax[sort_index[10:50]])
                income_tax_100 = np.mean(self.eval_env.income_tax[sort_index[50:]])
            
                total_wealth = np.mean(self.eval_env.households.at_next)
                wealth_10 = np.mean(self.eval_env.households.at_next[sort_index[:10]])
                wealth_50 = np.mean(self.eval_env.households.at_next[sort_index[10:50]])
                wealth_100 = np.mean(self.eval_env.households.at_next[sort_index[50:]])
            
                wealth_tax = np.mean(self.eval_env.asset_tax)
                wealth_tax_10 = np.mean(self.eval_env.asset_tax[sort_index[:10]])
                wealth_tax_50 = np.mean(self.eval_env.asset_tax[sort_index[10:50]])
                wealth_tax_100 = np.mean(self.eval_env.asset_tax[sort_index[50:]])
            
                per_gdp = self.eval_env.per_household_gdp
                income_gini = self.eval_env.income_gini
                wealth_gini = self.eval_env.wealth_gini
                wage = self.eval_env.WageRate
            
                total_labor = self.eval_env.Lt
                labor = self.eval_env.households.e * self.eval_env.ht
                labor_10 = np.mean(labor[sort_index[:10]])
                labor_50 = np.mean(labor[sort_index[10:50]])
                labor_100 = np.mean(labor[sort_index[50:]])
            
                sw = np.mean(house_reward)
                sw_10 = np.mean(house_reward[sort_index[:10]])
                sw_50 = np.mean(house_reward[sort_index[10:50]])
                sw_100 = np.mean(house_reward[sort_index[50:]])
            
                total_consumption = np.mean(self.eval_env.consumption)
                consumption_10 = np.mean(self.eval_env.consumption[sort_index[:10]])
                consumption_50 = np.mean(self.eval_env.consumption[sort_index[10:50]])
                consumption_100 = np.mean(self.eval_env.consumption[sort_index[50:]])
            
                # todo add new indicators
                datas = [gov_reward, sw, steps, total_income, income_10, income_50, income_100, total_tax, income_tax,
                         income_tax_10, income_tax_50, income_tax_100, total_wealth,
                         wealth_10, wealth_50, wealth_100, wealth_tax, wealth_tax_10, wealth_tax_50, wealth_tax_100,
                         per_gdp, income_gini, wealth_gini, wage, total_labor, labor_10,
                         labor_50, labor_100, sw_10, sw_50, sw_100, total_consumption, consumption_10, consumption_50,
                         consumption_100]
            
                episode_indicators.append(datas)
                if step_count == 1 or step_count == 100 or step_count == 200 or step_count == 300:
                    save_parameters(self.model_path, step_count, epoch_i, self.eval_env)
                if done:
                    break
            
                global_obs = next_global_obs
                private_obs = next_private_obs
            years = len(episode_indicators)
            avg_episode_data = np.mean(episode_indicators, axis=0)
            avg_episode_data[:3] *= years
            economic_indicators.append(avg_episode_data)

        return economic_indicators

    def light_episode_evolution(self):
        '''
        当面对相同的 state information，
        测试 MADDPG，random，fixed policy 分别做出的 action 导致的 tax,utility,labor,comsumption 分别是多少
        '''
        
    
        for epoch_i in range(self.args.eval_episodes):
            step_count = 0
            global_obs, private_obs = self.eval_env.reset()
            global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)

            maddpg_data = [[] for i in range(4)]
            random_data = [[] for i in range(4)]
            fixed_data = [[] for i in range(4)]
        
            while True:
                with torch.no_grad():
                    action, sort_index = self._evaluate_get_action(global_obs, private_obs)
                    
                    self.random_env = copy.copy(self.eval_env)
                    self.fixed_env = copy.copy(self.eval_env)
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)
                    
                    # todo 设计其他 policy 的动作 random policy + fixed policy
                    random_action = self.random_evaluate_get_action(action)
                    _,_,_,random_house_rew, _ = self.random_env.step(random_action)
                    fix_action = self.fixed_evaluate_get_action(self.fixed_env, action)
                    _,_,_,fixed_house_rew, _ = self.fixed_env.step(fix_action)

                # todo 统计数据
                step_count += 1
                
                # 对 10%， 10～50%， 50%～100% 人群分组 找到三组的平均数据
                maddpg_data[0].append(np.mean(self.eval_env.tax_array))
                maddpg_data[1].append(np.mean(house_reward))
                maddpg_data[2].append(self.eval_env.Lt)
                maddpg_data[3].append(np.mean(self.eval_env.consumption))

                random_data[0].append(np.mean(self.random_env.tax_array))
                random_data[1].append(np.mean(random_house_rew))
                random_data[2].append(self.random_env.Lt)
                random_data[3].append(np.mean(self.random_env.consumption))

                fixed_data[0].append(np.mean(self.fixed_env.tax_array))
                fixed_data[1].append(np.mean(fixed_house_rew))
                fixed_data[2].append(self.fixed_env.Lt)
                fixed_data[3].append(np.mean(self.fixed_env.consumption))

                
                if done:
                    break
            
                global_obs = next_global_obs
                private_obs = next_private_obs
            if step_count > 298:
                # 指定要保存的文件名
                self.sav_list(str(self.model_path) + "/maddpg_episode_"+str(epoch_i)+".csv", maddpg_data)
                self.sav_list(str(self.model_path) + "/random_episode_"+str(epoch_i)+".csv", random_data)
                self.sav_list(str(self.model_path) + "/fixed_episode_"+str(epoch_i)+".csv", fixed_data)

    def sav_list(self, file_name, data_list):
        
        # 打开CSV文件并将数据写入
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in data_list:
                writer.writerow(row)
    
    def _evaluate_get_action(self, global_obs, private_obs):
        global_obs_tensor = self._get_tensor_inputs(global_obs)
        private_obs_tensor = self._get_tensor_inputs(private_obs)
        hou_action = np.zeros((self.envs.households.n_households, self.args.house_action_dim))
        gov_action = self.agents[-1].select_action(global_obs_tensor, self.noise, self.epsilon)
    
        n_global_obs = global_obs_tensor.repeat(self.envs.households.n_households, 1)
        obs = torch.cat([n_global_obs, private_obs_tensor],
                        dim=-1)
        # 根据wealth 排序observation
        sorted_indices = torch.argsort(obs[:, -1], descending=True)
        sorted_obs = obs[sorted_indices]
        num_set = range(0, self.envs.households.n_households)
        for i in range(self.args.agent_block_num - 1):
            if i == 0:
                num = num_set[:int(0.1 * self.envs.households.n_households)]
            elif i == 1:
                num = num_set[int(0.1 * self.envs.households.n_households):int(0.5 * self.envs.households.n_households)]
            else:
                num = num_set[int(0.5 * self.envs.households.n_households):]
            hou_action[num] = self.agents[i].select_action(sorted_obs[num], self.noise, self.epsilon)
        # temp = np.random.random((self.args.n_households, self.envs.households.action_space.shape[1]))
        # hou_action = temp * 2 - 1
        # todo 按照原顺序指定 households actions
        house_sort_index = sorted_indices.cpu().numpy()
        hou_action = hou_action[np.argsort(house_sort_index)]
    
        action = {self.envs.government.name: gov_action,
                  self.envs.households.name: hou_action}
        
        return action, house_sort_index

    def random_evaluate_get_action(self, original_action):
        gov_action = original_action[self.envs.government.name]
    
        temp = np.random.random((self.args.n_households, self.envs.households.action_space.shape[1]))
        hou_action = temp * 2 - 1
        action = {self.envs.government.name: gov_action,
                  self.envs.households.name: hou_action}
    
        return action

    def fixed_evaluate_get_action(self, env,original_action):
        gov_action = original_action[self.envs.government.name]
        IFE =2
        CRRA = 1
        m = (IFE / (IFE + CRRA)) * np.log((IFE / (IFE + CRRA)) * np.exp(0.045))
        e = 0.200 * np.random.random((self.args.n_households,1))
        h = 1/2 * e - 1/2 * m
        c = np.log(env.households.e) - e + m
        h = np.exp(h) / 2
        c = np.exp(c)
    
        temp = np.random.random((self.args.n_households, self.envs.households.action_space.shape[1]))
        temp[:, 0] = 1-c.flatten()
        temp[:, 1] = h.flatten()
        hou_action = temp * 2 - 1
        action = {self.envs.government.name: gov_action,
                  self.envs.households.name: hou_action}
    
        return action
