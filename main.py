import numpy as np

from env.env_core import economic_society
from agents.baseline import agent
from utils.seeds import set_seeds
from arguments import get_args
import os

if __name__ == '__main__':
    # set signle thread
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # get arguments
    args = get_args()
    set_seeds(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    # todo start to create the environment
    #
    # if args.env_type == 'atari':
    #     envs = create_multiple_envs(args)
    # elif args.env_type == 'mujoco':
    #     envs = create_single_env(args)
    # else:
    #     raise NotImplementedError

    # todo test
    env = economic_society(args)

    print(env.households.e_transition(1))

    # todo trainer 包括 government + N households
    trainer = agent(env, args)
    # start to learn
    trainer.learn()
    # close the environment
    env.close()
