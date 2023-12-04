
from torch.distributions.normal import Normal


def select_actions(pi):
    mean, std = pi
    actions = Normal(mean, std).sample()
    # return actions
    return actions.detach().cpu().numpy().squeeze()

def evaluate_actions(pi, actions):
    mean, std = pi
    normal_dist = Normal(mean, std)
    log_prob = normal_dist.log_prob(actions).sum(dim=1, keepdim=True)
    entropy = normal_dist.entropy().mean()
    return log_prob, entropy
