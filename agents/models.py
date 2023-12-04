import torch
import torch.nn as nn
import torch.nn.functional as F

class CloneModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(CloneModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_mean = nn.Linear(hidden_size, output_size)
        self.fc_std = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = F.sigmoid(self.fc_mean(x))  # Apply sigmoid activation
        log_std = torch.clamp(self.fc_std(x), min=-20, max=2)
        std = torch.exp(log_std)  # Ensure std is positive
        return mean, std

class Critic(nn.Module):
    def __init__(self, input_dims, hidden_size, action_dims=None):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size) if action_dims is None else nn.Linear(input_dims + action_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)

        self.initialize_weights()
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

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


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.1)
        nn.init.constant_(m.bias, 0)
class Actor(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_size, log_std_min, log_std_max):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        # self.gru = nn.GRU(128, 256, 1, batch_first=True)
        self.fc2 = nn.Linear(128, 128)
        self.tanh = nn.Tanh()
        self.mean = nn.Linear(128, action_dims)
        self.log_std = nn.Linear(128, action_dims)
        self.log_std_min = log_std_min
        self.log_std_max = 1
        self.mean_max = 1
        self.mean_min = -1

    def forward(self, obs):
        out = F.relu(self.fc1(obs))
        out = F.relu(self.fc2(out))
        # out = self.tanh(out)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        mean = torch.clamp(mean, min=self.mean_max, max=self.mean_min)

        return (mean, torch.exp(log_std))


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
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


class mlp(nn.Module):
    def __init__(self, input_size, output_size, num_agent, log_std_min, log_std_max):
        super(mlp, self).__init__()
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

    def forward(self, global_state, private_state, update=False):
        n_global_obs = global_state.repeat(1, self.num_agent, 1)

        inputs = torch.cat([n_global_obs, private_state], dim=-1)
        out = self.fc1(inputs)
        out = self.gru(out)
        out = self.fc2(out)
        out = self.tanh(out)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return (mean, torch.exp(log_std))

class MFSharedAgent(nn.Module):
    def __init__(self, input_size, gov_action_dim, house_action_dim, num_agent, log_std_min, log_std_max):
        super(MFSharedAgent, self).__init__()
        self.fc1 = nn.Linear(input_size+gov_action_dim+house_action_dim, 64)
        self.gru = GRU(64, 128, 1, 0.1)
        self.fc2 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()
        self.mean = nn.Linear(64, house_action_dim)
        self.log_std = nn.Linear(64, house_action_dim)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_agent = num_agent

    def forward(self, global_state, private_state, gov_action, past_mean_house_action, update=False):
        if update == True:
            global_state = global_state.unsqueeze(1)
            gov_action = gov_action.unsqueeze(1)
            past_mean_house_action = past_mean_house_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)
        n_past_mean_house_action = past_mean_house_action.repeat(1, self.num_agent, 1)
        inputs = torch.cat([n_global_obs, private_state, n_gov_action, n_past_mean_house_action], dim=-1)
        out = self.fc1(inputs)
        out = self.gru(out)
        out = self.fc2(out)
        out = self.tanh(out)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return (mean, torch.exp(log_std))


import torch.nn as nn

class MFCritic(nn.Module):
    def __init__(self, input_dims, hidden_size, gov_action_dims, house_action_dim):
        super(MFCritic, self).__init__()
        self.fc1 = nn.Linear(input_dims + gov_action_dims + house_action_dim, hidden_size)
        # self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.ln2 = nn.LayerNorm(hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)


    def forward(self, obs, gov_action, mean_house_action):
        inputs = torch.cat([obs, gov_action, mean_house_action], dim=1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        output = self.q_value(x)
        return output


class MFSharedCritic(nn.Module):   # Q(s, a_g, a_h, \bar{a_h})
    def __init__(self, state_dim, gov_action_dim, hou_action_dim, hidden_size, num_agent):
        super(MFSharedCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + gov_action_dim + 2*hou_action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)
        self.num_agent = num_agent

    def forward(self, global_state, private_state, gov_action, hou_action, mean_house_action):

        global_state = global_state.unsqueeze(1)
        gov_action = gov_action.unsqueeze(1)
        mean_house_action = mean_house_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)
        n_mean_house_action = mean_house_action.repeat(1, self.num_agent, 1)

        inputs = torch.cat([n_global_obs, private_state, n_gov_action, hou_action, n_mean_house_action], dim=-1)  # 修改维度
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        output = self.q_value(x)
        return output


"""
this network also include gaussian distribution and beta distribution

"""
class mlp_net(nn.Module):
    def __init__(self, state_size, num_actions):
        super(mlp_net, self).__init__()
        self.fc1_v = nn.Linear(state_size, 64)
        self.fc2_v = nn.Linear(64, 64)
        self.fc1_a = nn.Linear(state_size, 64)
        self.fc2_a = nn.Linear(64, 64)

        self.sigma_log = nn.Parameter(torch.zeros(1, num_actions))
        self.action_mean = nn.Linear(64, num_actions)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.zero_()

        # define layers to output state value
        self.value = nn.Linear(64, 1)
        self.value.weight.data.mul_(0.1)
        self.value.bias.data.zero_()

    def forward(self, x):
        x_v = torch.tanh(self.fc1_v(x))
        x_v = torch.tanh(self.fc2_v(x_v))
        state_value = self.value(x_v)
        # output the policy...
        x_a = torch.tanh(self.fc1_a(x))
        x_a = torch.tanh(self.fc2_a(x_a))

        mean = self.action_mean(x_a)
        sigma_log = self.sigma_log.expand_as(mean)
        sigma = torch.exp(sigma_log)
        pi = (mean, sigma)

        return state_value, pi


class BMF_actor(nn.Module):
    def __init__(self, input_size, gov_action_dim, house_action_dim, num_agent, log_std_min, log_std_max):
        super(BMF_actor, self).__init__()
        self.fc1 = nn.Linear(input_size+gov_action_dim+house_action_dim*2, 64)  # pi(observation, gov_a, top10_action, bot50_action)
        self.gru = GRU(64, 128, 1, 0.1)
        self.fc2 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()
        self.mean = nn.Linear(64, house_action_dim)
        self.log_std = nn.Linear(64, house_action_dim)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_agent = num_agent

    def forward(self, global_state, private_state, gov_action, past_mean_house_action, update=False):
        if update == True:
            global_state = global_state.unsqueeze(1)
            gov_action = gov_action.unsqueeze(1)
            past_mean_house_action = past_mean_house_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)
        n_past_mean_house_action = past_mean_house_action.repeat(1, self.num_agent, 1)
        inputs = torch.cat([n_global_obs, private_state, n_gov_action, n_past_mean_house_action], dim=-1)
        out = self.fc1(inputs)
        out = self.gru(out)
        out = self.fc2(out)
        out = self.tanh(out)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return (mean, torch.exp(log_std))

# class BMF_actor_1(nn.Module):  # pi(s)  没有 mean action
#     def __init__(self, input_size, gov_action_dim, house_action_dim, num_agent, log_std_min, log_std_max):
#         super(BMF_actor_1, self).__init__()
#         self.fc1 = nn.Linear(input_size+gov_action_dim, 128)  # pi(observation, gov_a, top10_action, bot50_action)
#         self.gru = GRU(128, 256, 1, 0.1)
#         self.fc2 = nn.Linear(256, 128)
#         self.tanh = nn.Tanh()
#         self.mean = nn.Linear(128, house_action_dim)
#         self.log_std = nn.Linear(128, house_action_dim)
#         # the log_std_min and log_std_max
#         self.log_std_min = log_std_min
#         self.log_std_max = log_std_max
#         self.num_agent = num_agent
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, global_state, private_state, gov_action, update=False):
#         if update == True:
#             global_state = global_state.unsqueeze(1)
#             gov_action = gov_action.unsqueeze(1)
#
#         n_global_obs = global_state.repeat(1, self.num_agent, 1)
#         n_gov_action = gov_action.repeat(1, self.num_agent, 1)
#         inputs = torch.cat([n_global_obs, private_state, n_gov_action], dim=-1)
#         out = self.fc1(inputs)
#         out = self.gru(out)
#         out = self.fc2(out)
#         out = self.tanh(out)
#         mean = self.mean(out)
#         log_std = self.log_std(out)
#         log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
#
#         return (mean, torch.exp(log_std))
class BMF_actor_1(nn.Module):
    def __init__(self, input_size, gov_action_dim, house_action_dim, num_agent, log_std_min, log_std_max):
        super(BMF_actor_1, self).__init__()
        self.fc1 = nn.Linear(input_size + gov_action_dim, 128)

        self.fc2 = nn.Linear(128, 128)
        self.tanh = nn.Tanh()
        self.mean = nn.Linear(128, house_action_dim)
        self.log_std = nn.Linear(128, house_action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_agent = num_agent
 

    def forward(self, global_state, private_state, gov_action, update=False):
        if update:
            global_state = global_state.unsqueeze(1)
            gov_action = gov_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)
        inputs = torch.cat([n_global_obs, private_state, n_gov_action], dim=-1)
        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        out = F.relu(out)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        mean = torch.clamp(mean, min=-1, max=1)

        return (mean, torch.exp(log_std))


class BMF_critic(nn.Module):
    def __init__(self, state_dim, gov_action_dim, hou_action_dim, hidden_size, num_agent):
        super(BMF_critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + gov_action_dim + 3*hou_action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)
        self.num_agent = num_agent


    def forward(self, global_state, private_state, gov_action, hou_action, mean_house_action):
        global_state = global_state.unsqueeze(1)
        gov_action = gov_action.unsqueeze(1)
        mean_house_action = mean_house_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)
        n_mean_house_action = mean_house_action.repeat(1, self.num_agent, 1)

        inputs = torch.cat([n_global_obs, private_state, n_gov_action, hou_action, n_mean_house_action], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        output = self.q_value(x)
        return output
