import matplotlib.pyplot as plt

algorithms = ['Algorithm 1', 'Algorithm 2', 'Algorithm 3', 'Algorithm 4', 'Algorithm 5']
gdp_values = [10, 8, 12, 9, 11]
error_values = [1, 0.5, 1.2, 0.8, 1.1]

bars = plt.bar(algorithms, gdp_values, yerr=error_values, capsize=5)

# 为每个柱形对象设置标签
for bar in bars:
    bar.set_label(bar.get_height())

# 显示图例
plt.legend()

plt.title('GDP by Algorithm')
plt.xlabel('Algorithms')
plt.ylabel('GDP')

plt.show()
