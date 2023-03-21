import torch
import torch.nn as nn
import torch.nn.functional as F


# # define the policy network - tanh gaussian policy network
# for government
class Actor(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_size, log_std_min, log_std_max):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dims)
        self.log_std = nn.Linear(hidden_size, action_dims)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        # clamp the log std
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return (mean, torch.exp(log_std))

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out

# for households
class SharedAgent(nn.Module):
    def __init__(self, input_size, output_size, num_agent, log_std_min, log_std_max):
        super(SharedAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.gru = GRU(64, 128, 1, 0.1)
        self.fc2 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()
        self.mean = nn.Linear(64, output_size)
        self.log_std = nn.Linear(64, output_size)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_agent = num_agent

    def forward(self, global_state, private_state, gov_action, update=False):
        if update == True:
            global_state = global_state.unsqueeze(1)
            gov_action = gov_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)
        inputs = torch.cat([n_global_obs, private_state, n_gov_action], dim=-1)
        out = self.fc1(inputs)
        out = self.gru(out)
        out = self.fc2(out)
        out = self.tanh(out)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return (mean, torch.exp(log_std))


class Critic(nn.Module):
    def __init__(self, input_dims, hidden_size, action_dims=None):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size) if action_dims is None else nn.Linear(input_dims + action_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)

    def forward(self, obs, action=None):
        inputs = torch.cat([obs, action], dim=1) if action is not None else obs
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        output = self.q_value(x)
        return output

class SharedCritic(nn.Module):   # Q(s, a_g, a_h, \bar{a_h})
    def __init__(self, state_dim, hou_action_dim, hidden_size, num_agent):
        super(SharedCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + hou_action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)
        self.num_agent = num_agent

    def forward(self, global_state, private_state, gov_action, hou_action):

        global_state = global_state.unsqueeze(1)
        gov_action = gov_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)

        inputs = torch.cat([n_global_obs, private_state, n_gov_action, hou_action], dim=-1)  # 修改维度
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        output = self.q_value(x)
        return output
