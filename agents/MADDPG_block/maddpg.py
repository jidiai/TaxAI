import torch
import os
from .actor_critic import Actor, Critic
import numpy as np
import copy

class MADDPG:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        
        critic_input_size = self.args.gov_action_dim + self.args.house_action_dim * self.args.n_households + self.args.gov_obs_dim + (self.args.house_obs_dim-self.args.gov_obs_dim) * self.args.n_households

        # create the network
        if agent_id == self.args.agent_block_num-1:   # government agent
            self.actor_network = Actor(args.gov_obs_dim, args.gov_action_dim, hidden_size=args.hidden_size)
            self.actor_target_network = Actor(args.gov_obs_dim, args.gov_action_dim, hidden_size=args.hidden_size)
        else:  # household agent
            self.actor_network = Actor(args.house_obs_dim, args.house_action_dim, hidden_size=args.hidden_size)
            self.actor_target_network = Actor(args.house_obs_dim, args.house_action_dim, hidden_size=args.hidden_size)
        
        self.critic_network = Critic(critic_input_size, hidden_size=args.hidden_size)
        self.critic_target_network = Critic(critic_input_size, hidden_size=args.hidden_size)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.p_lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.q_lr)
        
        # if use the cuda...
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            if self.agent_id == self.args.agent_block_num-1:
                action_dim = self.args.gov_action_dim
            else:
                action_dim = self.args.house_action_dim
            u = np.random.uniform(-1, 1, action_dim)
        else:
            # inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.actor_network(o).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.detach().cpu().numpy()
            noise = noise_rate * np.random.rand(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -1, +1)
        return u.copy()

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
    
    def households_obs_sort(self, private_obs):
        sorted_indices = torch.argsort(private_obs[:,:,-1], descending=True)
        return sorted_indices
    
    # update the network
    def train(self, transitions, other_agents):
        global_obs, private_obs, gov_action, hou_action, gov_reward, house_reward, next_global_obs, next_private_obs, done = transitions
        
        global_obses = torch.tensor(global_obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        private_obses = torch.tensor(private_obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_actions = torch.tensor(gov_action, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        hou_actions = torch.tensor(hou_action, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_rewards = torch.tensor(gov_reward, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        house_rewards = torch.tensor(house_reward, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_global_obses = torch.tensor(next_global_obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_private_obses = torch.tensor(next_private_obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - done, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)

        num_set = range(0, self.args.n_households)
        num = []
        num.append(num_set[:int(0.1 * self.args.n_households)])
        num.append(num_set[int(0.1 * self.args.n_households):int(0.5 * self.args.n_households)])
        num.append(num_set[int(0.5 * self.args.n_households):])

        sorted_index = self.households_obs_sort(private_obses)
        private_obses = private_obses[np.arange(self.args.batch_size)[:, None], sorted_index]
        hou_actions = hou_actions[np.arange(self.args.batch_size)[:, None], sorted_index]
        house_rewards = house_rewards[np.arange(self.args.batch_size)[:, None], sorted_index]
        next_private_obses = next_private_obses[np.arange(self.args.batch_size)[:, None], sorted_index]
        # inverse_dones = inverse_dones[np.arange(self.args.batch_size)[:, None], sorted_index]
        
        if self.agent_id == self.args.agent_block_num - 1:  # government agent
            r = gov_rewards.view(-1, 1)
        else:
            r = house_rewards[:, num[self.agent_id]]
  
        o = torch.cat((private_obses.reshape(self.args.batch_size, -1), global_obses), dim=-1)
        u = torch.cat((hou_actions.reshape(self.args.batch_size, -1), gov_actions), dim=-1)
        n_next_global_obses = next_global_obses.unsqueeze(1).repeat(1, self.args.n_households, 1)
        o_next = torch.cat((next_private_obses,n_next_global_obses), dim=-1)

        # calculate the target Q value function
        u_next = []
        this_next_o = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.agent_block_num):
                if agent_id == self.args.agent_block_num - 1:
                    this_next_o.append(next_global_obses)
                else:
                    this_next_o.append(o_next[:, num[agent_id]])
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(this_next_o[agent_id]).reshape(self.args.batch_size, -1))
                else:
                    u_next.append(other_agents[index].actor_target_network(this_next_o[agent_id]).reshape(self.args.batch_size, -1))
                    index += 1
            u_next = torch.cat(u_next, dim=1)
            flatten_o_next = torch.cat((next_private_obses.reshape(self.args.batch_size,-1),next_global_obses), dim=-1)
            q_next = (self.critic_target_network(flatten_o_next, u_next).detach()) * inverse_dones
            if self.agent_id == self.args.agent_block_num - 1:
                target_q = (r + self.args.gamma * q_next).detach()
            else:
                target_q = (r + self.args.gamma * q_next.unsqueeze(2).repeat(1,len(num[self.agent_id]),1)).detach()

        # the q loss
        if self.agent_id == self.args.agent_block_num - 1:
            q_value = self.critic_network(o, u)
        else:
            q_value = self.critic_network(o, u).unsqueeze(2).repeat(1,len(num[self.agent_id]),1)
        critic_loss = (target_q - q_value).pow(2).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # the actor loss
        new_house_actions = copy.copy(hou_actions)
        new_gov_actions = copy.copy(gov_actions)
        if self.agent_id == self.args.agent_block_num - 1:
            new_gov_actions = self.actor_network(global_obses)
        else:
            this_o = torch.cat((global_obses.unsqueeze(1).repeat(1, len(num[self.agent_id]),1), private_obses[:, num[self.agent_id]]), dim=-1)
            new_house_actions[:, num[self.agent_id]] = self.actor_network(this_o)
        
        u = torch.cat((new_house_actions.view(self.args.batch_size, -1), new_gov_actions), dim=1)
        actor_loss = - self.critic_network(o, u).mean()
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


        self._soft_update_target_network()
        self.train_step += 1
        
        return actor_loss.item(), critic_loss.item()

