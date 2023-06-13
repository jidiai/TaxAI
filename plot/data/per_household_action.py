import numpy as np
import pickle
import os
import matplotlib as mpl
# font_1 = {'family':'Arial',
#         'size': 18}
# font = {'family': 'Arial',
#         'weight': 'normal',
#         'size': 14}
font_1 = {'size': 18}
font = {'weight': 'normal',
        'size': 12}

mpl.rc('font', **font)
def load_params_from_file(filename):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    return params

# file_path = "/home/mqr/code/AI-TaxingPolicy/agents/models/independent_ppo/100/ppo/epoch_0_step_1_100_gdp_parameters.pkl"
# para = load_params_from_file(file_path)
# print("ppo_action:", para["government"])
# /home/mqr/code/AI-TaxingPolicy/agents/models/bmfac/100/run6/epoch_0_step_1_100_social_welfare_parameters.pkl
def fetch_data(alg):
    gov_action = []
    households_action = []
    households_utilty = []
    households_at = []
    for i in range(5):
        # path = "/home/mqr/code/AI-TaxingPolicy/agents/models/independent_ppo/10/"+ alg+"/epoch_0_step_%d_10_gdp_parameters.pkl"%(i+1)
        path = "/home/mqr/code/AI-TaxingPolicy/agents/models/independent_ppo/100/"+ alg+"/epoch_0_step_%d_100_gdp_parameters.pkl"%(i+1)
        # path = "/home/mqr/code/AI-TaxingPolicy/agents/models/bmfac/100/"+ alg+"/epoch_0_step_%d_100_social_welfare_parameters.pkl"%(i+1)
        para = load_params_from_file(path)
        gov_action.append([para['government'].tau, para['government'].xi, para['government'].tau_a, para['government'].xi_a, para['Gt_prob']])
        households_action.append([para['workingHours'], para["saving_p"]])
        households_utilty.append([para['households_reward']])
        households_at.append([para['households'].at])
    return gov_action, households_action, households_utilty, households_at

import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = [1,2,3,4,5] # 横坐标

# ppo_data = fetch_data("run11")
# ppo_data = fetch_data("run7")
ppo_data = fetch_data("run63")
num_agent = 100
data1 = np.array(ppo_data[0]).T

ppo_h = np.array(ppo_data[1])[:,0]
ppo_p = np.array(ppo_data[1])[:,1]
ppo_reward = np.array(ppo_data[2])[:,0]
house_at = np.array(ppo_data[3])[:,0]


# 对第二维度进行排序，返回排序索引
sorted_indices = np.broadcast_to(np.argsort(-house_at[0], axis=0), (5, num_agent, 1))

# 根据排序索引对数组进行排序
house_at = np.take_along_axis(house_at, sorted_indices, axis=1)
ppo_reward = np.take_along_axis(ppo_reward, sorted_indices, axis=1)
ppo_p = np.take_along_axis(ppo_p, sorted_indices, axis=1)
ppo_h = np.take_along_axis(ppo_h, sorted_indices, axis=1)


# 设置图形大小
fig, axes = plt.subplots(10,5, figsize=(20, 28))
colors = ['#264653','#2A9D8F','#E9C46A','#F4A261','#E76F51']

green = '#2A9D8F'
yellow = '#E9C46A'
# 第一张子图
# gov_action_label = ['tau', 'xi', 'tau_a', 'xi_a', "G/GDP"]
step_list = ["1",'2','3','4','5']
# for i in range(5):
#     axes[0,0].plot(x, data1[i], marker='o', linestyle='-', label=gov_action_label[i], color=colors[i])
#
# axes[0,0].set_title('a) Government Action',font=font_1)
# axes[0,0].set_ylabel('Tax & Spending',font=font_1)
# axes[0,0].set_xlabel('Step',font=font_1)
# # axes[0,0].set_xlabel('Step')
# axes[0,0].set_xticks(np.arange(min(x), max(x)+1, 1))
# axes[0,0].legend(prop=font)

x = np.arange(5)
bar_width = 0.3  # bar的宽度
bar_int = 0.08


def sub_action_plot(index):
    if index > 49:
        x_index = (index-50) // 5
        y_index = (index-50) % 5
    else:
        x_index = (index) // 5
        y_index = (index) % 5
    bar_3 = axes[x_index,y_index].bar(x, ppo_h[:,index].flatten(), width=bar_width, align='center', color=yellow, label='Working')
    # ax3 = axes[x_index, y_index].twinx()
    bar_4 = axes[x_index,y_index].bar(x + bar_width+bar_int, ppo_p[:,index].flatten(), width=bar_width, align='center', color=green, label="Saving")
    ax4 = axes[x_index, y_index].twinx()
    line_2 = ax4.plot(x, ppo_reward[:,index].flatten(), marker='o', color='#E76F51', linestyle='dashed', label="Social Welfare=%d"%(np.sum(ppo_reward[:,index])))
    if y_index == 4:
        ax4.set_ylabel('Lifetime Utility',font=font_1)
    axes[x_index,y_index].set_title('IPPO Households %d'%(index+1))
    axes[x_index,0].set_ylabel('Working & Saving',font=font_1)
    axes[9,y_index].set_xlabel('Step',font=font_1)
    axes[x_index,y_index].set_xticks(x + (bar_width+bar_int) / 2)
    axes[x_index,y_index].set_xticklabels(step_list, fontdict=font_1)
    axes[x_index,y_index].set_ylim(0, 1)
    lines = [bar_3, bar_4]
    labels = [line.get_label() for line in lines]
    axes[0, 0].legend(loc='center left', prop=font)
    # ax3.legend( prop=font)
    # ax3.set_ylim(-100, 1000)
    ax4.legend(loc='upper right', prop=font)
    
 
for i in range(50):
    sub_action_plot(i+50)
    # sub_action_plot(i)

#
# bar_5 = axes[2].bar(x, ga_h, width=bar_width, align='center', color=yellow, label='Working')
# bar_6 = axes[2].bar(x + bar_width+bar_int, ga_p, width=bar_width, align='center', color=green, label="Saving")
# ax4 = axes[2].twinx()
# line_3 = ax4.plot(x, ga_reward, marker='o', color='#E76F51', linestyle='dashed',label="Social Welfare=%d"%(np.sum(ga_reward)))
# # ax4.set_ylabel('Social Welfare',font=font_1)
# axes[2].set_title('c) GA Households')
# # axes[2].set_ylabel('Working & Saving Probability',font=font_1)
# axes[2].set_xlabel('Step',font=font_1)
# axes[2].set_xticks(x + (bar_width+bar_int) / 2)
# axes[2].set_xticklabels(step_list, fontdict=font_1)
# axes[2].set_ylim(0, 1)
# lines = [bar_5, bar_6]
# labels = [line.get_label() for line in lines]
# # axes[2].legend(lines, labels, loc='upper left', prop=font)
# ax4.set_ylim(-100, 1000)
# ax4.legend(loc='upper right', prop=font)

# axes[0,0].grid(axis='y')
# axes[1].grid(axis='y')
# axes[2].grid(axis='y')
# axes[3].grid(axis='y')



# 调整子图之间的间距
fig.align_ylabels()
plt.subplots_adjust(hspace=0.1)
plt.tight_layout()
# plt.grid()
plt.savefig("tax_action_bottom50.pdf")
# plt.savefig("tax_action_top50.pdf")
# 显示图形
plt.show()

