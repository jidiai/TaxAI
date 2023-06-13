import numpy as np
import matplotlib.pyplot as plt

'''================================= data ===================================='''
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
yss = []
for _ in range(num_seeds):
    y = np.random.normal(loc=np.cos(x), scale=0.5)
    yss.append(y)
    
'''================================= plot ===================================='''
# 创建一个4x4的子图
fig, axes = plt.subplots(4, 4, figsize=(16, 12))


def plot_curves(data, ax, color, label):
    # 假设data是包含训练曲线数据的二维数组
    mean_curve = np.mean(data, axis=0)
    variance_curve = np.var(data, axis=0)
    # 假设window_size是平滑窗口的大小
    window_size = 50
    smooth_mean_curve = np.convolve(mean_curve, np.ones(window_size), 'valid') / window_size
    smooth_variance_curve = np.convolve(variance_curve, np.ones(window_size), 'valid') / window_size
    # 绘制均值曲线和方差曲线
    ax.plot(smooth_mean_curve,color=color, label=label)
    ax.fill_between(range(len(smooth_variance_curve)), smooth_mean_curve - np.sqrt(smooth_variance_curve), smooth_mean_curve + np.sqrt(smooth_variance_curve), color=color, alpha=0.3)
    


# 在每个子图中绘制不同的曲线
for i in range(4):
    for j in range(4):
        ax = axes[i, j]  # 获取当前子图
        plot_curves(ys, ax=ax, color='b', label='IPPO')
        plot_curves(yss, ax=ax, color='red', label="MADDPG")
        ax.legend()  # 显示图例
        ax.set_title('Training Curve')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid()
        


'''================================= label ===================================='''


# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
