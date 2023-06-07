import numpy as np
import pickle
import os
import matplotlib as mpl
# font_1 = {'family':'Arial',
#         'size': 18}
# font = {'family': 'Arial',
#         'weight': 'normal',
#         'size': 14}
font_1 = {'size': 20}
font = {'weight': 'normal',
        'size': 15}

mpl.rc('font', **font)
def load_params_from_file(filename):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    return params

# file_path = "/home/mqr/code/AI-TaxingPolicy/agents/models/independent_ppo/100/ppo/epoch_0_step_1_100_gdp_parameters.pkl"
# para = load_params_from_file(file_path)
# print("ppo_action:", para["government"])

def fetch_data(alg):
    gov_action = []
    households_action = []
    households_utilty = []
    for i in range(5):
        path = "/home/mqr/code/AI-TaxingPolicy/agents/models/independent_ppo/100/"+ alg+"/epoch_0_step_%d_100_gdp_parameters.pkl"%(i+1)
        para = load_params_from_file(path)
        gov_action.append([para['government'].tau, para['government'].xi, para['government'].tau_a, para['government'].xi_a, para['Gt_prob']])
        households_action.append([np.mean(para['workingHours']), para["mean_saving_p"]])
        households_utilty.append([np.sum(para['households_reward']), np.sum(para['post_income'])])
    return gov_action, households_action, households_utilty

import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = [1,2,3,4,5] # 横坐标
# data1 = np.random.randint(1, 10, size=(5, 5))  # 第一张子图的数据

random_data = fetch_data("run26")
ppo_data = fetch_data("run28")
ga_data = fetch_data("run38")

data1 = np.array(ppo_data[0]).T
random_h = np.array(random_data[1])[:,0]
random_p = np.array(random_data[1])[:,1]
random_reward = np.array(random_data[2])[:,0]

ppo_h = np.array(ppo_data[1])[:,0]
ppo_p = np.array(ppo_data[1])[:,1]
ppo_reward = np.array(ppo_data[2])[:,0]

ga_h = np.array(ga_data[1])[:,0]
ga_p = np.array(ga_data[1])[:,1]
ga_reward = np.array(ga_data[2])[:,0]

# 设置图形大小
fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
colors = ['#264653','#2A9D8F','#E9C46A','#F4A261','#E76F51']

green = '#2A9D8F'
yellow = '#E9C46A'
# 第一张子图
gov_action_label = ['tau', 'xi', 'tau_a', 'xi_a', "G/GDP"]
step_list = ["1",'2','3','4','5']
for i in range(5):
    axes[0].plot(x, data1[i], marker='o', linestyle='-', label=gov_action_label[i], color=colors[i])

axes[0].set_title('a) Government Action',font=font_1)
axes[0].set_ylabel('Tax & Spending',font=font_1)
axes[0].set_xlabel('Step',font=font_1)
# axes[0].set_xlabel('Step')
axes[0].set_xticks(np.arange(min(x), max(x)+1, 1))
axes[0].legend(prop=font)

x = np.arange(5)
# 第二张子图
bar_width = 0.3  # bar的宽度
bar_int = 0.08
bar_1 = axes[1].bar(x, random_h, width=bar_width, align='center', color=yellow, label='Working')
bar_2 = axes[1].bar(x + bar_width+bar_int, random_p, width=bar_width, align='center', color=green, label="Saving")
ax2 = axes[1].twinx()
line_1 = ax2.plot(x, random_reward, marker='o', color='#E76F51', linestyle='dashed', label="Social Welfare=%d"%(np.sum(random_reward)))
# ax2.set_ylabel('Social Welfare',font=font_1)
# axes[1].text(1, 1, 'sw=%d'%(np.sum(random_reward)), font=font)
axes[1].set_title('b) Random Households',font=font_1)
axes[1].set_ylabel('Working & Saving',font=font_1)
axes[1].set_xlabel('Step',font=font_1)
axes[1].set_xticks(x + (bar_width+bar_int) / 2)
axes[1].set_xticklabels(step_list, fontdict=font_1)
axes[1].set_ylim(0, 1)
lines = [bar_1, bar_2]
labels = [line.get_label() for line in lines]
axes[1].legend(lines, labels, loc='upper left', prop=font)
ax2.set_ylim(-100, 1000)
ax2.legend(loc='center left',bbox_to_anchor=(0., 0.65), prop=font)


# 第三张子图

bar_3 = axes[3].bar(x, ppo_h, width=bar_width, align='center', color=yellow, label='Working')
bar_4 = axes[3].bar(x + bar_width+bar_int, ppo_p, width=bar_width, align='center', color=green, label="Saving")
ax3 = axes[3].twinx()
line_2 = ax3.plot(x, ppo_reward, marker='o', color='#E76F51', linestyle='dashed', label="Social Welfare=%d"%(np.sum(ppo_reward)))
ax3.set_ylabel('Social Welfare',font=font_1)
axes[3].set_title('d) IPPO Households')
# axes[3].set_ylabel('Working & Saving Probability',font=font_1)
axes[3].set_xlabel('Step',font=font_1)
axes[3].set_xticks(x + (bar_width+bar_int) / 2)
axes[3].set_xticklabels(step_list, fontdict=font_1)
axes[3].set_ylim(0, 1)
lines = [bar_3, bar_4]
labels = [line.get_label() for line in lines]
# axes[3].legend(lines, labels, loc='upper left', prop=font)
ax3.set_ylim(-100, 1000)
ax3.legend(loc='upper right', prop=font)

bar_5 = axes[2].bar(x, ga_h, width=bar_width, align='center', color=yellow, label='Working')
bar_6 = axes[2].bar(x + bar_width+bar_int, ga_p, width=bar_width, align='center', color=green, label="Saving")
ax4 = axes[2].twinx()
line_3 = ax4.plot(x, ga_reward, marker='o', color='#E76F51', linestyle='dashed',label="Social Welfare=%d"%(np.sum(ga_reward)))
# ax4.set_ylabel('Social Welfare',font=font_1)
axes[2].set_title('c) GA Households')
# axes[2].set_ylabel('Working & Saving Probability',font=font_1)
axes[2].set_xlabel('Step',font=font_1)
axes[2].set_xticks(x + (bar_width+bar_int) / 2)
axes[2].set_xticklabels(step_list, fontdict=font_1)
axes[2].set_ylim(0, 1)
lines = [bar_5, bar_6]
labels = [line.get_label() for line in lines]
# axes[2].legend(lines, labels, loc='upper left', prop=font)
ax4.set_ylim(-100, 1000)
ax4.legend(loc='upper right', prop=font)

axes[0].grid(axis='y')
axes[1].grid(axis='y')
axes[2].grid(axis='y')
axes[3].grid(axis='y')



# 调整子图之间的间距
plt.subplots_adjust(hspace=0.4)
plt.tight_layout()
# plt.grid()
plt.savefig("tax_action.pdf")
# 显示图形
plt.show()

