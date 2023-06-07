import numpy as np
import matplotlib.pyplot as plt

font = {'family':'Arial',
        'size': 18}
font_1 = {'family':'Arial',
        'size': 12}

wealth_data = np.loadtxt("data/wealth_stack.txt")
time = np.arange(len(wealth_data))
# 绘制百分比堆叠面积图
labels = ['top 10%', '10~20%', '20~30%', '30~50%', 'bottom 50%']
colors = ['#FFC107', '#FF9800', '#FF5722', '#E91E63', '#9C27B0']

fig, ax = plt.subplots()
ax.stackplot(time, wealth_data.T, labels=labels, colors=colors)
ax.legend(loc='upper left', prop=font_1)

# 设置图形标题和轴标签
plt.title('Wealth Distribution on Free Market Policy', font=font)
plt.xlabel('epochs', font=font)
plt.ylabel('Share of Wealth', font=font)
plt.tight_layout()

plt.show()
