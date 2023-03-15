import numpy as np
import math
import random
import quantecon as qe
import matplotlib.pyplot as plt

x = np.linspace(0.01, 1, 1000)

def pareto(x):
    a = 5
    return np.power(x, -1/a)

y = pareto(x)
y_new = (y-min(y))/(max(y)-min(y))
fig, ax = plt.subplots()
ax.plot(x, y_new, label='Pareto distribution')


ax.legend()
plt.show()