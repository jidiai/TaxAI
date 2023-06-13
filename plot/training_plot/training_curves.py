import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as ticker
font = {'family':'Arial',
        'size': 18}
font_1 = {'family':'Arial',
        'size': 13}
'''================================= data ===================================='''

'''================================= free & GA data ===================================='''

import pickle


def load_params_from_file(filename):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    return params


def fetch_free_data():
    free = []
    for i in range(9):
        path = "/home/mqr/code/AI-TaxingPolicy/agents/models/rule_based/100/run40/epoch_%d_step_1_100_gdp_parameters.pkl" % (
                    i + 1)
        if os.path.exists(path):
            para = load_params_from_file(path)
            free.append([para['step_cnt'], para['government_reward'], np.mean(para['households_reward']), para['per_household_gdp'],para['wealth_gini'],para['income_gini']])
    return np.mean(free,axis=0)

def fetch_ga_data(goal):
    ga = []
    for i in range(10):
        path = "/home/mqr/code/AI-TaxingPolicy/agents/models/ga/run2/epoch_%d_step_200_"%i+"100_"+ goal+"_parameters.pkl"
        if os.path.exists(path):
            para = load_params_from_file(path)
            ga.append([para['step_cnt'], para['government_reward'], np.mean(para['households_reward']), para['per_household_gdp'],para['wealth_gini'],para['income_gini']])
    return np.mean(ga,axis=0)
free_data = fetch_free_data()
ga_gdp = fetch_ga_data("gdp")
ga_gini = fetch_ga_data("gini")
ga_sw = fetch_ga_data("social_welfare")
ga_mix = fetch_ga_data("gdp_gini")
ga = [ga_gdp, ga_gini, ga_sw, ga_mix]


folder_path = '/home/mqr/code/AI-TaxingPolicy/plot/training_plot/'
'''================================= data ===================================='''
maddpg_gdp = []
ippo_gdp = []
bmfac_gdp = []
def gdp_append_data(data_frame):
    data_array = data_frame.values
    maddpg_gdp.append([data_array[:, 1][:1500], data_array[:, 1 + 3][:1500]])
    ippo_gdp.append([data_array[:, 1 + 3 * 2][:1500], data_array[:, 1 + 3 * 4][:1500]])
    bmfac_gdp.append([data_array[:, 1 + 3 * 3][:1500]])

# 获取文件夹中的所有文件名称
file_names = os.listdir(folder_path + 'gdp/')
file_names = sorted(file_names)
# 打印文件名称
for file_name in file_names:
    gdp_append_data(pd.read_csv(folder_path + 'gdp/' + file_name))

maddpg_gini = []
ippo_gini = []
bmfac_gini = []
def gini_append_data(data_frame):
    data_array = data_frame.values
    maddpg_gini.append([data_array[:, 1][:1500], data_array[:, 1 + 3][:1500], data_array[:, 1 + 3*5][:1500]])
    ippo_gini.append([data_array[:, 1 + 3 * 2][:1500], data_array[:, 1 + 3 * 4][:1500]])
    bmfac_gini.append([data_array[:, 1 + 3 * 3][:1500]])

# 获取文件夹中的所有文件名称
file_names = os.listdir(folder_path + 'gini/')
file_names = sorted(file_names)
# 打印文件名称
for file_name in file_names:
    gini_append_data(pd.read_csv(folder_path + 'gini/'+file_name))

maddpg_sw = []
ippo_sw = []
bmfac_sw = []
def sw_append_data(data_frame):
    data_array = data_frame.values
    maddpg_sw.append([data_array[:, 1 +3*2][:1500], data_array[:, 1 + 3][:1500], data_array[:, 1 + 3*5][:1500]])
    ippo_sw.append([data_array[:, 1][:1500], data_array[:, 1 + 3 * 4][:1500]])
    bmfac_sw.append([data_array[:, 1 + 3 * 3][:1500]])


# 获取文件夹中的所有文件名称
file_names = os.listdir(folder_path + 'sw/')
file_names = sorted(file_names)
# 打印文件名称
for file_name in file_names:
    sw_append_data(pd.read_csv(folder_path + 'sw/'+file_name))
#
maddpg_mix = []
ippo_mix = []
bmfac_mix = []
def mix_append_data(data_frame):
    data_array = data_frame.values
    maddpg_mix.append([data_array[:, 1 + 3*3][:1500], data_array[:, 1 + 3*4][:1500], data_array[:, 1 + 3*5][:1500]])
    ippo_mix.append([data_array[:, 1 ][:1500], data_array[:, 1 + 3 * 2][:1500]])
    # bmfac_mix.append([data_array[:, 1 + 3 ]])

# 获取文件夹中的所有文件名称
file_names = os.listdir(folder_path + 'mix/')
file_names = sorted(file_names)
# 打印文件名称
for file_name in file_names:
    mix_append_data(pd.read_csv(folder_path + 'mix/'+file_name))


'''================================= txt data ===================================='''

bmfac_mix = [[], [],[],[],[],[]]
txt_path = "/home/mqr/code/AI-TaxingPolicy/agents/models/bmfac/run"

files_label = ["/years.txt","/gov_reward.txt","/house_reward.txt", "/gdp.txt", "/wealth_gini.txt", "/income_gini.txt"]
def fetch_txt_data(index,data_list):
    path = txt_path + str(index)
    for i in range(6):
        data = np.loadtxt(path + files_label[i])
        data_list[i].append(data)

fetch_txt_data(5, bmfac_gini)
fetch_txt_data(7, bmfac_sw)
# fetch_txt_data(6, bmfac_mix)
fetch_txt_data(8, bmfac_mix)

'''================================= data ===================================='''


maddpg = [maddpg_gdp, maddpg_gini, maddpg_sw, maddpg_mix]
ippo = [ippo_gdp, ippo_gini, ippo_sw, ippo_mix]
bmfac = [bmfac_gdp, bmfac_gini, bmfac_sw, bmfac_mix]
# maddpg = [maddpg_gdp, maddpg_gini, maddpg_gini, maddpg_gini]
# ippo = [ippo_gdp, ippo_gini, ippo_gini, ippo_gini]
# bmfac = [bmfac_gdp, bmfac_gini, ippo_gini, ippo_gini]

'''================================= plot ===================================='''
# 创建一个4x4的子图
fig, axes = plt.subplots(6, 4, figsize=(16, 18))
# colors = ['#264653','#1A6158','#D5A220','#E57010','#A23216']
# colors = ['#264653','#E9C46A','#2A9D8F','#F4A261','#E76F51']
colors = ['#6A4C93','#1982C4','#8AC926','#E76F51','#F4A261']

steps = np.arange(500,1000000,500)
def plot_curves(data, ax, color, label):
    # 假设data是包含训练曲线数据的二维数组
    mean_curve = np.mean(data, axis=0)
    variance_curve = np.var(data, axis=0)
    # 假设window_size是平滑窗口的大小
    window_size = 100
    smooth_mean_curve = np.convolve(mean_curve, np.ones(window_size), 'valid') / window_size
    smooth_variance_curve = np.convolve(variance_curve, np.ones(window_size), 'valid') / window_size
    # 绘制均值曲线和方差曲线
    ax.plot(steps[:len(smooth_mean_curve)], smooth_mean_curve, color=color, label=label)
    ax.fill_between(steps[:len(smooth_mean_curve)], smooth_mean_curve - np.sqrt(smooth_variance_curve),
                    smooth_mean_curve + np.sqrt(smooth_variance_curve), color=color, alpha=0.2)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useOffset=True)


labels =  ["Years", "Government returns", "Social welfare", "Per capital GDP", "Wealth gini", "Income gini"]
tasks = ['Max GDP', 'Min Inequality', 'Max Social Welfare', 'Multi-Tasks']

# 每一条数据 代表所有seed 在当前task，当前指标下的数据
for i in range(6):  # row 每一行指不同的指标
    for j in range(4):   # column 每一列指 不同的task
        ax = axes[i, j]  # 获取当前子图
        x_steps = np.arange(500, steps[1500], 75000)
        free = free_data[i] * np.ones(10)
        # ax.axhline(free_data[i], linestyle='dashed', color=colors[0], label="Free")
        # ax.axhline(free_data[i], marker='^', linestyle='dashed', color=colors[0], label="Free")
        ax.plot(x_steps, free, marker='^', linestyle='dashed', color=colors[0], label="Free")
        ax.plot(x_steps, ga[j][i]*np.ones(10), marker='*', linestyle='dashed',color=colors[1], label="GA")
        # ax.axhline(ga[j][i], marker='*', linestyle='dashed',color=colors[1], label="GA")
        # ax.axhline(ga[j][i], linestyle='dashed', color=colors[1], label="GA")
        plot_curves(ippo[j][i], ax=ax, color=colors[2], label="IPPO")
        plot_curves(maddpg[j][i], ax=ax, color=colors[3], label='MADDPG')
        plot_curves(bmfac[j][i], ax=ax, color=colors[4], label="BMFAC")
        axes[0, 3].legend(prop=font_1)  # 显示图例
        if i == 0:
            ax.set_title(tasks[j], font=font)
        if i == 5:
            ax.set_xlabel('Steps', font=font)
        if j == 0:
            ax.set_ylabel(labels[i], font=font)
        ax.grid()

'''================================= label ===================================='''

# 调整子图之间的间距
fig.align_ylabels()
plt.tight_layout()
plt.savefig("training_curves.pdf")
# 显示图形
plt.show()
