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

    env = economic_society(args)
    env.reset()
    done = False
    next_agent_name = 'government'
    action_value = None
    step = 0

    while not done:
        if env.agent_selection == 'households':
            action_value = np.random.random(size=(5, 2))
        elif env.agent_selection == 'government':
            action_value = np.random.random(size=(5,))

        action = {env.agent_selection: action_value}
        obs, r, done, next_agent_name, _ = env.step(action)  # o^{-i}_{t+1}, r^i_t(s,a), agent_name^{-i}
        print("step ",step, "reward:", r, "--done: ", done, "--next agent: ", next_agent_name)
        step += 1


    # # todo trainer 包括 government + N households
    # trainer = agent(env, args)
    # # start to learn
    # trainer.learn()
    # # close the environment
    # env.close()
