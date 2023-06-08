import matplotlib.pyplot as plt
import numpy as np
# 存储 不同策略不同steps的asset即可

wealth_set = np.loadtxt("data/wealth_set_100_free.txt")
font_1 = {'family':'Arial',
        'size': 15}
def gini_coef(wealths):
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / (len(cum_wealths) - 1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    gini_coef = A / (A + B)
    return xarray, yarray, gini_coef

COLOR = ["b", "orange", "green","red","purple"]
def plot_lorenz(xarray, yarray, gini,alg, ax, i, color):
    # Plotting the Lorenz curve
    ax.plot(xarray, yarray, color=color, label='%s'%alg)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Line of perfect equality
    ax.fill_between(xarray, yarray, alpha= 0.2,color="lightblue")
    

    


def plot_gini(col,ax, font, wealth_set,colors):
    algorithms = ['Free', 'GA', 'IPPO', 'MADDPG', 'BMFAC']
    for i in range(len(wealth_set)):
        if wealth_set[i] is not 0:
            x, y, gini = gini_coef(wealth_set[i].reshape(-1, 1))
            plot_lorenz(x, y, gini, algorithms[i], ax[1, col], i, color=colors[i])
    
    ax[1, col].set_xlabel('Cumulative Share of Population', font=font_1)
    ax[1, 0].set_ylabel('Cumulative Share of Wealth', font=font)
    ax[1, 0].legend(prop=font_1)
    ax[1, col].grid(axis='y')
    
    
