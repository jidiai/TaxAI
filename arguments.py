import argparse

# define the arguments that will be used in the SAC
def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--env-name', type=str, default='wealth_distribution', help='the environment name')
    parse.add_argument('--device-num', type=int, default=1, help='the number of cuda service num')
    # parse.add_argument('--total-timesteps', type=int, default=int(1e7), help='the total num of timesteps to be run')
    parse.add_argument('--cuda', action='store_true', help='use GPU do the training')
    parse.add_argument('--seed', type=int, default=123, help='the random seed to reproduce results')
    parse.add_argument('--hidden-size', type=int, default=256, help='the size of the hidden layer')
    # parse.add_argument('--train-loop-per-epoch', type=int, default=1, help='the training loop per epoch')
    parse.add_argument('--q-lr', type=float, default=3e-4, help='the learning rate')
    parse.add_argument('--p-lr', type=float, default=3e-4, help='the learning rate of the actor')
    parse.add_argument('--n-epochs', type=int, default=int(3e3), help='the number of total epochs')
    parse.add_argument('--epoch-length', type=int, default=int(1e3), help='the lenght of each epoch')
    # parse.add_argument('--n-updates', type=int, default=int(1e3), help='the number of training updates execute')
    parse.add_argument('--init-exploration-steps', type=int, default=int(1e3), help='the steps of the initial exploration')
    parse.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the replay buffer')
    parse.add_argument('--init-exploration-policy', type=str, default='gaussian', help='the inital exploration policy')
    parse.add_argument('--batch-size', type=int, default=64, help='the batch size of samples for training')
    # parse.add_argument('--reward-scale', type=float, default=1, help='the reward scale')
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parse.add_argument('--log-std-max', type=float, default=2, help='the maximum log std value')
    parse.add_argument('--log-std-min', type=float, default=-20, help='the minimum log std value')
    parse.add_argument('--target-update-interval', type=int, default=1, help='the interval to update target network')
    parse.add_argument('--update-cycles', type=int, default=int(500), help='how many updates apply in the update')
    parse.add_argument('--eval-episodes', type=int, default=10, help='the episodes that used for evaluation')
    parse.add_argument('--tau', type=float, default=5e-3, help='the soft update coefficient')
    parse.add_argument('--display-interval', type=int, default=1, help='the display interval')
    parse.add_argument('--consumption_range', type=float, default=1.0, help='the max term of consumption')
    parse.add_argument('--working_hours_range', type=float, default=24.0, help='the max term of working hours')
    parse.add_argument('--saving_prob', type=float, default=1.0, help='Proportion of savings')  # 有budget constraint, saving与consumption只用一个即可
    parse.add_argument('--CRRA', type=float, default=2.0, help='CRRA(Coefficient of Relative Risk Aversion)')
    parse.add_argument('--IFE', type=float, default=0.6, help='Inverse Frisch elasticity')
    # parse.add_argument('--beta', type=float, default=0.96, help='discount factor')
    parse.add_argument('--alpha', type=float, default=1/3, help='capitial elasticity')
    parse.add_argument('--delta', type=float, default=0.05, help='depreciation rate')
    parse.add_argument('--captial_rental_rate', type=float, default=0.04, help='captial_rental_rate')
    parse.add_argument('--lump_sum_transfer', type=float, default=0.0001, help='lump sum transfer')
    parse.add_argument('--initial_e', type=float, default=int(1), help='e_0, initial abilities')
    parse.add_argument('--initial_debt', type=float, default=1.0, help='the intial government debt')  # todo?
    parse.add_argument('--initial_wealth', type=float, default=1.0, help='the initial wealth distribution')  # todo????
    parse.add_argument('--n_households', type=int, default=5, help='the number of total households')




    return parse.parse_args()
