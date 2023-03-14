import numpy as np
import random
import torch
import os

# set random seeds for the pytorch, numpy and random
def set_seeds(args, rank=0):
    # set seeds for the numpy
    np.random.seed(args.seed + rank)
    # set seeds for the random.random
    random.seed(args.seed + rank)
    os.environ['PYTHONHASHSEED'] = str(args.seed + rank)
    # set seeds for the pytorch
    torch.manual_seed(args.seed + rank)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

