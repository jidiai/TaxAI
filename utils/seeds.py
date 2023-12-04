import numpy as np
import random
import torch
import os

# set random seeds for the pytorch, numpy and random
def set_seeds(seed,cuda=False):
    # set seeds for the numpy
    np.random.seed(seed)
    # set seeds for the random.random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # set seeds for the pytorch
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

