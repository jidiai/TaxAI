import numpy as np
from scipy.optimize import minimize


def calculate_h(W, e, I):
    def objective(h):
        L = np.dot(e, h)
        return np.abs(W - (1 - alpha) * (K / L) ** alpha)

    alpha = 0.3  # 可以调整的参数
    K = 1000  # 已知的资本存量

    bounds = [(0, 1) for _ in range(len(e))]
    h0 = np.ones(len(e)) / len(e)
    res = minimize(objective, h0, bounds=bounds)
    return res.x * np.sum(e * I) / np.sum(e * res.x)


# 生成一些假数据
W = 1.2
e = np.random.rand(10)
I = np.random.rand(10)

# 计算 h
h = calculate_h(W, e, I)
print(h)  # 输出结果
