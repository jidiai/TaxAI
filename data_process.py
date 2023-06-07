import numpy as np
import pickle
import os

def load_params_from_file(filename):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    return params

# parameters = load_params_from_file("/home/mqr/code/AI-TaxingPolicy/agents/models/rule_based/10/run30/epoch_0_step_1_10_gdp_parameters.pkl")

free_path = "/home/mqr/code/AI-TaxingPolicy/agents/models/rule_based/10/run30/epoch_%d_step_1_10_gdp_parameters.pkl"
def data_process(path):
    parameters = []
    gdp = []
    wealth_set = []
    sw = []
    gini = []
    num = 0
    
    for i in range(10):
        file_path = path % i
        if os.path.exists(file_path):
            num += 1

            para = load_params_from_file(file_path)
            parameters.append(para)
            gdp.append(para["per_household_gdp"])
            gini.append(para['wealth_gini'])
            wealth_set.append(para['households'].at)
            sw.append(para['households_reward'].sum())
        else:
            print(file_path + " don't exist!")
    
    gdp_mean = np.mean(gdp)
    gdp_std = np.std(gdp)
    
    gini_mean = np.mean(gini)
    gini_std = np.std(gini)
    
    sorted_arrays = []
    # 对每个数组进行排序
    for arr in wealth_set:
        sorted_arr = np.sort(arr)
        sorted_arrays.append(sorted_arr)
    # 将排序后的数组转换为NumPy数组
    sorted_arrays = np.array(sorted_arrays)
    # 计算排序后数组的均值
    mean_wealth_set = np.mean(sorted_arrays, axis=0)
    
    mean_sw = np.mean(sw)
    std_sw = np.std(sw)
    
    return [gdp_mean,gdp_std, gini_mean, gini_std, mean_wealth_set, mean_sw, std_sw]

step_list = [1, 100, 200, 300]
# print(data_process(free_path))
def bmfac(run_num, task,n):
    bmfac_path = "/home/mqr/code/AI-TaxingPolicy/agents/models/bmfac/"+str(n)+"/run%d"%run_num
    bmf_data = []
    for i in range(len(step_list)):
        file_path = bmfac_path + '/epoch_%d_step_' + str(step_list[i]) + '_' + str(n) + '_' + task + '_parameters.pkl'
        bmf_data.append(data_process(file_path))
    
    # print(bmf_data[0])
    # print(bmf_data[1])
    # print(bmf_data[2])
    return bmf_data


def free(run_num, task, n):
    bmfac_path = "/home/mqr/code/AI-TaxingPolicy/agents/models/rule_based/"+str(n)+"/run%d"%run_num
    # file_path = bmfac_path + '/epoch_%d_step_' + str(step_i) + '_' + str(n) +'_' + task +'_parameters.pkl'
    # step_list = [1]
    bmf_data = []
    for i in range(len(step_list)):
        file_path = bmfac_path + '/epoch_%d_step_' + str(step_list[i]) + '_' + str(n) + '_' + task + '_parameters.pkl'
        bmf_data.append(data_process(file_path))

    return bmf_data

def ga(run_num, task, n):
    bmfac_path = "/home/mqr/code/AI-TaxingPolicy/agents/models/ga/run%d"%run_num
    # file_path = bmfac_path + '/epoch_%d_step_' + str(step_i) + '_' + str(n) +'_' + task +'_parameters.pkl'
    step_list = [1, 100, 200, 300]
    bmf_data = []
    for i in range(len(step_list)):
        file_path = bmfac_path + '/epoch_%d_step_' + str(step_list[i]) + '_' + str(n) + '_' + task + '_parameters.pkl'
        # "/home/mqr/code/AI-TaxingPolicy/agents/models/ga/10/run1/epoch_0_step_10_100_social_welfare_parameters.pkl"
        bmf_data.append(data_process(file_path))

    return bmf_data

def maddpg(run_num, task, n):
    bmfac_path = "/home/mqr/code/AI-TaxingPolicy/agents/models/maddpg/"+str(n)+"/run%d" % run_num
    bmf_data = []
    for i in range(len(step_list)):
        file_path = bmfac_path + '/epoch_%d_step_' + str(step_list[i]) + '_' + str(n) + '_' + task + '_parameters.pkl'
        bmf_data.append(data_process(file_path))
    
    return bmf_data


def ppo(run_num, task, n):
    bmfac_path = "/home/mqr/code/AI-TaxingPolicy/agents/models/independent_ppo/" + str(n) + "/run%d" % run_num
    bmf_data = []
    for i in range(len(step_list)):
        file_path = bmfac_path + '/epoch_%d_step_' + str(step_list[i]) + '_' + str(n) + '_' + task + '_parameters.pkl'
        bmf_data.append(data_process(file_path))
    
    return bmf_data

free_data = free(32,"gdp",10)
# print("{:.3f}".format(free_data))
maddpg_gdp = maddpg(14, "gdp", 100)
maddpg_gini = maddpg(13, "gini", 100)
# todo maddpg_sw

# todo bmfac_gdp

# todo bmfac_gini

# todo bmfac_sw

# todo ppo_gdp
# todo ppo_gini
# todo ppo_sw

# todo ga_gdp
# todo ga_gini
# todo ga_sw


#
# print(free_data)
# print(maddpg_gdp)
# print(maddpg_gini)

