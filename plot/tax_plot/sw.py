import pandas as pd
import numpy as np
import math

font_1 = {'family':'Arial',
        'size': 15}
'''============================================================================================'''


def plot_sw(col, axes, font, error_attri, gdp_values,error_values,colors):
    bar_width = 0.6
    # 五个算法的名称
    algorithms = ['Free', 'GA', 'IPPO', 'MADDPG', 'BMFAC']
    #
    # # 每个算法对应的 GDP 值
    # gdp_values = [10, 8, 12, 9, 11]
    #
    # # 每个算法对应的误差范围（可以根据实际情况调整）
    # error_values = [1, 0.5, 1.2, 0.8, 1.1]
    
    # 绘制柱形图
    for i in range(len(algorithms)):
        axes[2, col].bar(algorithms[i], gdp_values[i],bar_width,label=algorithms[i], yerr=error_values[i], error_kw=error_attri[i][0], color=colors[i])
    axes[2, col].set_xticklabels(algorithms, fontdict=font_1, rotation=30)
    axes[2, 0].set_ylim(-1000, 1300)
    # 添加标题和标签
    # axes[0, col].title('GDP by Algorithm')
    # axes[0, 3].xlabel('Algorithms')
    axes[2, 0].set_ylabel('Social Welfare', font=font)
    
    # axes[2, 3].legend(prop=font_1,loc='upper left' )
    axes[2, col].grid(axis='y')