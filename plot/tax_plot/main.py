from gdp_bar import plot_gdp
from lorenz import plot_gini
from sw import plot_sw
from tax import plot_tax
import numpy as np
import matplotlib.pyplot as plt

font = {'family':'Arial',
        'size': 20}
font_1 = {'family':'Arial',
        'size': 18}


# 生成一些示例数据
'''============================================================================================'''

from data_process import free,ga,maddpg,bmfac,ppo


# free_data = free(32,"gdp",10)
free_data = free(40,"gdp",100)
ga_gdp_data = ga(2,"gdp",100)
ga_gini_data = ga(2,"gini",100)
ga_sw_data = ga(2,"social_welfare",100)
# print("{:.3f}".format(free_data))
maddpg_gdp = maddpg(24, "gdp", 100)# 14
maddpg_gini = maddpg(25, "gini", 100)
maddpg_sw = maddpg(26, "social_welfare", 100)

# ppo_gdp_10 = ppo(29, "gdp", 10)
ppo_gdp_100 = ppo(29, "gdp", 100)
ppo_gini_100 = ppo(30, "gini", 100)
ppo_sw_100 = ppo(31, "social_welfare", 100)

bmfac_gdp_100 = bmfac(23, "gdp",100)
bmfac_gini_100 = bmfac(24, "gini",100)
bmfac_sw_100 = bmfac(22, "social_welfare",100)
# ppo_gini = ppo(2, "gini", 10)
# todo GA
# gdp
def data(free, ga, ppo, maddpg,bmfac, index):
    data = []
    for i in range(len(free)):
        sub_list = [free[i][index], ga[i][index], ppo[i][index], maddpg[i][index], bmfac[i][index]]
        for j in range(len(sub_list)):
            if np.isnan(np.mean(sub_list[j])):
                sub_list[j] = 0
        data.append(sub_list)
    return data

gdp_values = data(free_data, ga_gdp_data, ppo_gdp_100, maddpg_gdp, bmfac_gdp_100, index=0)
error_values = data(free_data, ga_gdp_data, ppo_gdp_100, maddpg_gdp, bmfac_gdp_100, index=1)

wealth_set = data(free_data, ga_gini_data, ppo_gini_100, maddpg_gini, bmfac_gini_100, index=4)

sw_mean = data(free_data, ga_sw_data, ppo_sw_100, maddpg_sw, bmfac_sw_100, index=5)
sw_std = data(free_data, ga_sw_data, ppo_sw_100, maddpg_sw, bmfac_sw_100, index=6)

'''============================================================================================'''
# 创建一个4x4子图的大图
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 10))

steps = [1,100,200,300]

# colors = ['#F1FAEE','#E63946', '#A8DADC','#457B9D','#1D3557']
# colors = ['#E76F51','#F4A261', '#E9C46A','#2A9D8F','#264653']
colors = ['#264653','#2A9D8F','#E9C46A','#F4A261','#E76F51']
colors_line = ['#264653','#1A6158','#D5A220','#E57010','#A23216']
plt.rcParams['font.family'] = 'Times New Roman'

error_attri = [[dict(elinewidth=1,ecolor=colors_line[j],capsize=5)] for j in range(5)]
# 第一行 - GDP柱形图
for i in range(len(steps)):
    plot_gdp(i, axes, font=font_1, error_attri = error_attri, gdp_values =gdp_values[i], error_values=error_values[i],colors=colors)


for i in range(len(steps)):
    plot_gini(i,axes,font=font_1, wealth_set=wealth_set[i],colors=colors_line)
    
for i in range(len(steps)):
    plot_sw(i, axes, font=font_1, error_attri = error_attri, gdp_values =sw_mean[i], error_values=sw_std[i],colors=colors)


# 调整子图之间的间距
fig.align_ylabels()
plt.subplots_adjust(hspace=0.4)
plt.tight_layout()
# plt.legend(prop=font_1)
plt.savefig("results.pdf")
# 显示图形

plt.show()
