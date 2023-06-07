import pandas as pd
import numpy as np
import math
'''============================================================================================'''


def plot_tax(col, axes, font, error_attri):

    bar_width = 0.6
    # 五个算法的名称
    algorithms = ['Free', 'ABM', 'IPPO', 'MADDPG', 'BMFAC']
    n = len(algorithms)
    tax_rate_data = np.random.rand(n)
    tax_data = np.random.rand(n)
    
    # 每个算法对应的误差范围（可以根据实际情况调整）
    error_values = np.random.rand(n) * 0.1
    
    # 绘制柱形图
    for i in range(len(algorithms)):
        axes[3, col].bar(algorithms[i], tax_data[i], bar_width,label=algorithms[i], yerr=error_values[i], error_kw=error_attri)
        
    axes[3, col].plot(range(n), tax_rate_data, marker='o', linestyle='--', color='r', label="tax rate")
    axes[3, 0].set_ylabel('Tax Revenue and Tax Rate', font=font)
    axes[3, 3].legend(prop=font)
    axes[3, col].grid(axis='y')
