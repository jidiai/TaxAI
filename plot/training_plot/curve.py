import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以获得可重复的随机数据
np.random.seed(0)

# 生成随机数据
x = np.linspace(0, 10, 1000)
num_seeds = 5  # 随机种子的数量

# 生成每个随机种子对应的y值
ys = []
for _ in range(num_seeds):
    y = np.random.normal(loc=np.sin(x), scale=0.2)
    ys.append(y)

# 计算y的平均值、最大值和最小值
y_mean = np.mean(ys, axis=0)
y_max = np.max(ys, axis=0)
y_min = np.min(ys, axis=0)

# 平滑处理曲线
y_mean_smooth = np.convolve(y_mean, np.ones(300) / 300, mode='same')

# 绘制曲线图
plt.plot(x, y_mean_smooth, color='blue', label='Mean')  # 平均值曲线
plt.fill_between(x, y_min, y_max, color='blue', alpha=0.3, label='Min-Max Range')  # 最小值和最大值的阴影区域

# 添加图例和标题
plt.legend()
plt.title('Random Seed Curve')

# 显示图形
plt.show()
