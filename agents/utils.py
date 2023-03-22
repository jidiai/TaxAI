import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions import Distribution
from torch.nn import functional as F

"""
the tanhnormal distributions from rlkit may not stable

"""
class tanh_normal(Distribution):
    def __init__(self, normal_mean, normal_std, epsilon=1e-6, cuda=False):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.cuda = cuda
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        # value = value * 2 - 1  # 因为我做了一个 tanh放（0，1）的操作
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        sample_mean = torch.zeros(self.normal_mean.size(), dtype=torch.float32, device='cuda' if self.cuda else 'cpu')
        sample_std = torch.ones(self.normal_std.size(), dtype=torch.float32, device='cuda' if self.cuda else 'cpu')
        z = (self.normal_mean + self.normal_std * Normal(sample_mean, sample_std).sample())
        z.requires_grad_()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

# get action_infos
class get_action_info:
    def __init__(self, pis, cuda=False):
        self.mean, self.std = pis
        self.dist = tanh_normal(normal_mean=self.mean, normal_std=self.std, cuda=cuda)

    # select actions
    def select_actions(self, exploration=True, reparameterize=True):
        if exploration:
            if reparameterize:
                actions, pretanh = self.dist.rsample(return_pretanh_value=True)
                return torch.sigmoid(actions), pretanh
            else:
                actions = self.dist.sample()
        else:
            actions = torch.tanh(self.mean)
        return torch.sigmoid(actions)

    def get_log_prob(self, actions, pre_tanh_value):
        log_prob = self.dist.log_prob(actions, pre_tanh_value=pre_tanh_value)
        return log_prob.sum(dim=1, keepdim=True)

#
# import numpy as np
# import torch
# from torch.distributions.normal import Normal
# from torch.distributions import Distribution
# from torch.nn import functional as F
#
# """
# the sigmoidnormal distributions from rlkit may not stable
#
# """
#
#
# class sigmoid_normal(Distribution):
#     def __init__(self, normal_mean, normal_std, epsilon=1e-6, cuda=False):
#         self.normal_mean = normal_mean
#         self.normal_std = normal_std
#         self.cuda = cuda
#         self.normal = Normal(normal_mean, normal_std)
#         self.epsilon = epsilon
#
#     def sample_n(self, n, return_pre_sigmoid_value=False):
#         z = self.normal.sample_n(n)
#         if return_pre_sigmoid_value:
#             return torch.sigmoid(z), z
#         else:
#             return torch.sigmoid(z)
#
#     def log_prob(self, value, pre_sigmoid_value=None):
#         """
#         :param value: some value, x
#         :param pre_sigmoid_value: arcsigmoid(x)
#         :return:
#         """
#         # value = value * 2 - 1  # 因为我做了一个 sigmoid放（0，1）的操作
#         if pre_sigmoid_value is None:
#             pre_sigmoid_value = 1/(1+torch.exp(-value))
#         return self.normal.log_prob(pre_sigmoid_value) - torch.log(1 - value * value + self.epsilon)
#
#     def sample(self, return_presigmoid_value=False):
#         """
#         Gradients will and should *not* pass through this operation.
#
#         See https://github.com/pytorch/pytorch/issues/4620 for discussion.
#         """
#         z = self.normal.sample().detach()
#         if return_presigmoid_value:
#             return torch.sigmoid(z), z
#         else:
#             return torch.sigmoid(z)
#
#     def rsample(self, return_presigmoid_value=False):
#         """
#         Sampling in the reparameterization case.
#         """
#         sample_mean = torch.zeros(self.normal_mean.size(), dtype=torch.float32, device='cuda' if self.cuda else 'cpu')
#         sample_std = torch.ones(self.normal_std.size(), dtype=torch.float32, device='cuda' if self.cuda else 'cpu')
#         z = (self.normal_mean + self.normal_std * Normal(sample_mean, sample_std).sample())
#         z.requires_grad_()
#         if return_presigmoid_value:
#             return torch.sigmoid(z), z
#         else:
#             return torch.sigmoid(z)
#
#
# # get action_infos
# class get_action_info:
#     def __init__(self, pis, cuda=False):
#         self.mean, self.std = pis
#         self.dist = sigmoid_normal(normal_mean=self.mean, normal_std=self.std, cuda=cuda)
#
#     # select actions
#     def select_actions(self, exploration=True, reparameterize=True):
#         if exploration:
#             if reparameterize:
#                 actions, presigmoid = self.dist.rsample(return_presigmoid_value=True)
#                 return torch.sigmoid(actions), presigmoid
#             else:
#                 actions = self.dist.sample()
#         else:
#             actions = torch.sigmoid(self.mean)
#         return actions
#
#     def get_log_prob(self, actions, pre_sigmoid_value):
#         log_prob = self.dist.log_prob(actions, pre_sigmoid_value=pre_sigmoid_value)
#         return log_prob.sum(dim=1, keepdim=True)
