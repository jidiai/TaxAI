import numpy as np
import pickle


def load_params_from_file(filename):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    return params

parameters = load_params_from_file("/home/mqr/code/AI-TaxingPolicy/agents/models/rule_based/10/run30/epoch_0_step_1_10_gdp_parameters.pkl")
at = parameters['households'].at
sw = parameters['households_reward'].sum()