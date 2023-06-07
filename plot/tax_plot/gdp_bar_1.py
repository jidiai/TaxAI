import pandas as pd
import numpy as np
import math


filename = 'data/0530_gdp.csv'  # CSV 文件的路径

data = pd.read_csv(filename)

data_name_list = list(data.axes[1])
data_values = data.values
plot_data_name_list = []
plot_data_list = []
for i in range(len(data_name_list)):
    if i % 3 == 1:
        plot_data_name_list.append(data_name_list[i])
        plot_data_list.append(data.values[:,i])
    
# ['rule_based-run236  n=100 - GDP', 'independent_ppo-run5  n=100 - GDP', 'independent_ppo-run4  n=10 - GDP', 'bmfac-run106  n=100 - GDP', 'maddpg-run36  n=100 - GDP', 'bmfac-run112  n=10 - GDP', 'rule_based-run235  n=10 - GDP', 'maddpg-run35  n=10 - GDP']
#
# data_dict = {k: v for k, v in zip(plot_data_name_list, plot_data_list)}

free_100 = pd.Series(list(plot_data_list[0])).dropna().tolist()
ppo_100 = pd.Series(list(plot_data_list[1])).dropna().tolist()
ppo_10 = pd.Series(list(plot_data_list[2])).dropna().tolist()
bmfac_100 = pd.Series(list(plot_data_list[3])).dropna().tolist()
maddpg_100 = pd.Series(list(plot_data_list[4])).dropna().tolist()
bmfac_10 = pd.Series(list(plot_data_list[5])).dropna().tolist()
free_10 = pd.Series(list(plot_data_list[6])).dropna().tolist()
maddpg_10 = pd.Series(list(plot_data_list[7])).dropna().tolist()

data_list = [free_100, ppo_100, ppo_10, bmfac_100, maddpg_100, bmfac_10, free_10, maddpg_10]
mean_list = []
for each in data_list:
    mean_list.append(np.mean(each[-100:]))
    
    
abm_gdp = 3650501.6927251206

import matplotlib.pyplot as plt
import numpy as np

font = {'family':'Arial',
        'size': 18}
font_1 = {'family':'Arial',
        'size': 12}
# 数据准备
algorithms = ['Free Market', 'Genetic Algorithm', 'Idependent PPO', 'MADDPG', 'BMFAC']
categories = ['Max GDP', 'Min Gini', 'Max Social Welfare', 'Multi-task']

# 每种算法的四类值（n=10）
values_algorithm_1_n10 = [10, 15, 12, 8]
values_algorithm_2_n10 = [7, 12, 10, 6]
values_algorithm_3_n10 = [9, 11, 14, 5]
values_algorithm_4_n10 = [13, 9, 8, 11]
values_algorithm_5_n10 = [6, 13, 9, 12]

# 每种算法的四类值（n=100）
values_algorithm_1_n100 = [15, 18, 14, 10]
values_algorithm_2_n100 = [10, 14, 12, 8]
values_algorithm_3_n100 = [12, 13, 16, 7]
values_algorithm_4_n100 = [16, 12, 10, 14]
values_algorithm_5_n100 = [9, 16, 12, 15]
std_err=[2,3,2.5,1.4]
error_attri = dict(elinewidth=1,ecolor="grey",capsize=3)
# 设置柱形图的宽度和间距
bar_width = 0.12
bar_spacing = 0.03

# 设置间隔
index = np.arange(len(categories))
index_shifted = [index + (bar_width + bar_spacing) * i for i in range(len(algorithms))]

# 创建上下两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# 绘制上面子图（n=10）
ax1.bar(index_shifted[0], values_algorithm_1_n10, bar_width, label='Free Market',yerr=std_err,error_kw=error_attri)
ax1.bar(index_shifted[1], values_algorithm_2_n10, bar_width, label='Genetic Algorithm',yerr=std_err,error_kw=error_attri)
ax1.bar(index_shifted[2], values_algorithm_3_n10, bar_width, label='Idependent PPO',yerr=std_err,error_kw=error_attri)
ax1.bar(index_shifted[3], values_algorithm_4_n10, bar_width, label='MADDPG',yerr=std_err,error_kw=error_attri)
ax1.bar(index_shifted[4], values_algorithm_5_n10, bar_width, label='BMFAC',yerr=std_err,error_kw=error_attri)

# 设置上面子图的标题和y轴标签
ax1.set_title('Comparison of the performance of different algorithms on 4 tasks (N = 10)', font=font)
ax1.set_ylabel('Per capita GDP', font=font)

# 添加网格线
ax1.grid(True)

# 绘制下面子图（n=100）
ax2.bar(index_shifted[0], values_algorithm_1_n100, bar_width, label='Free Market',yerr=std_err,error_kw=error_attri)
ax2.bar(index_shifted[1], values_algorithm_2_n100, bar_width, label='Genetic Algorithm',yerr=std_err,error_kw=error_attri)
ax2.bar(index_shifted[2], values_algorithm_3_n100, bar_width, label='Idependent PPO',yerr=std_err,error_kw=error_attri)
ax2.bar(index_shifted[3], values_algorithm_4_n100, bar_width, label='MADDPG',yerr=std_err,error_kw=error_attri)
ax2.bar(index_shifted[4], values_algorithm_5_n100, bar_width, label='BMFAC',yerr=std_err,error_kw=error_attri)

# 设置下面子图的标题和y轴标签
ax2.set_title('Comparison of the performance of different algorithms on 4 tasks (N = 100)', font=font)
ax2.set_ylabel('Per capita GDP', font=font)

# 添加网格线
ax2.grid(True)

# 设置x轴标签

plt.xlabel('Tasks', font=font)
# plt.suptitle('Comparison of the performance of different algorithms on 4 tasks')
plt.xticks(index + (bar_width + bar_spacing) * (len(algorithms) / 2), categories, font=font)
# 添加图例
ax1.legend(prop=font_1)
ax2.legend(prop=font_1)

# 调整子图间距
plt.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.savefig("gdp.pdf")
# 显示图形
plt.show()

