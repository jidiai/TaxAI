import pandas as pd
import numpy as np
import math


# filename = 'data/0530_gdp.csv'  # CSV 文件的路径
#
# data = pd.read_csv(filename)
#
# data_name_list = list(data.axes[1])
# data_values = data.values
# plot_data_name_list = []
# plot_data_list = []
# for i in range(len(data_name_list)):
#     if i % 3 == 1:
#         plot_data_name_list.append(data_name_list[i])
#         plot_data_list.append(data.values[:,i])
#
# # ['rule_based-run236  n=100 - GDP', 'independent_ppo-run5  n=100 - GDP', 'independent_ppo-run4  n=10 - GDP', 'bmfac-run106  n=100 - GDP', 'maddpg-run36  n=100 - GDP', 'bmfac-run112  n=10 - GDP', 'rule_based-run235  n=10 - GDP', 'maddpg-run35  n=10 - GDP']
# #
# # data_dict = {k: v for k, v in zip(plot_data_name_list, plot_data_list)}
#
# free_100 = pd.Series(list(plot_data_list[0])).dropna().tolist()
# ppo_100 = pd.Series(list(plot_data_list[1])).dropna().tolist()
# ppo_10 = pd.Series(list(plot_data_list[2])).dropna().tolist()
# bmfac_100 = pd.Series(list(plot_data_list[3])).dropna().tolist()
# maddpg_100 = pd.Series(list(plot_data_list[4])).dropna().tolist()
# bmfac_10 = pd.Series(list(plot_data_list[5])).dropna().tolist()
# free_10 = pd.Series(list(plot_data_list[6])).dropna().tolist()
# maddpg_10 = pd.Series(list(plot_data_list[7])).dropna().tolist()
#
# data_list = [free_100, ppo_100, ppo_10, bmfac_100, maddpg_100, bmfac_10, free_10, maddpg_10]
# mean_list = []
# for each in data_list:
#     mean_list.append(np.mean(each[-100:]))
#
#
# abm_gdp = 3650501.6927251206

font_1 = {'family':'Arial',
        'size': 15}
'''============================================================================================'''
steps = [1,100,200,300]

def plot_gdp(col,axes, font, error_attri, gdp_values,error_values, colors):
    bar_width = 0.6
    # 五个算法的名称
    algorithms = ['Free', 'GA', 'IPPO', 'MADDPG', 'BMFAC']
    
    # # 每个算法对应的 GDP 值
    # gdp_values = [10, 8, 12, 9, 11]
    #
    # # 每个算法对应的误差范围（可以根据实际情况调整）
    # error_values = [1, 0.5, 1.2, 0.8, 1.1]
    
    # 绘制柱形图
    for i in range(len(algorithms)):
        axes[0, col].bar(algorithms[i], gdp_values[i],bar_width,label=algorithms[i], yerr=error_values[i], error_kw=error_attri[i][0], color=colors[i])
    axes[0, col].set_title('Step=%d'%steps[col], font=font)
    axes[0, col].set_xticklabels(algorithms, fontdict=font_1, rotation=30)
    axes[0, col].set_ylim(0, 2e7)
    # 添加标题和标签
    # axes[0, col].title('GDP by Algorithm')
    # axes[0, 3].xlabel('Algorithms')
    axes[0, 0].set_ylabel('Per capita GDP', font=font)
    axes[0, 0].legend(loc='upper left',prop=font_1)
    axes[0, col].grid(axis='y')
    