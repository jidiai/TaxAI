from data_process import free,ga,maddpg,bmfac,ppo
import numpy as np

free_data_10 = free(35,"gdp",10)
free_data_100 = free(40,"gdp",100)
free_data_1000 = free(1,"gdp",1000)
free_data_10000 = free(1,"gdp",10000)
free = [free_data_10, free_data_100, free_data_1000,free_data_10000]

ga_gdp_10 = ga(2,"gdp",10)
ga_gdp_100 = ga(2,"gdp",100)
ga_data = [ga_gdp_10, ga_gdp_100]


maddpg_gdp_10 = maddpg(44, "gdp", 10)
maddpg_gdp_100 = maddpg(24, "gdp", 100)# 14
maddpg_gdp_1000 = maddpg(1, "gdp", 1000)# 14
maddpg_gdp_10000 = maddpg(1, "gdp", 10000)# 14
maddpg = [maddpg_gdp_10, maddpg_gdp_100, maddpg_gdp_1000, maddpg_gdp_10000]

ppo_gdp_10 = ppo(5, "gdp", 10)
ppo_gdp_100 = ppo(29, "gdp", 100)
ppo_gdp_1000 = ppo(4, "gdp", 1000)
ppo_gdp_10000 = ppo(4, "gdp", 10000)
# ppo_gdp_1000 = ppo(29, "gdp", 100)
ppo = [ppo_gdp_10, ppo_gdp_100, ppo_gdp_1000, ppo_gdp_10000]


bmfac_gdp_10 = bmfac(39, "gdp",10)
bmfac_gdp_100 = bmfac(23, "gdp",100)
bmfac_gdp_1000 = bmfac(1, "gdp",1000)
bmfac = [bmfac_gdp_10, bmfac_gdp_100, bmfac_gdp_1000]
# for j in range(len(free)):
#     for i in range(len(free[j])):
#         if np.isnan(free[j][i][0]):
#             break
#         mean = free[j][i][0]
#         std = free[j][i][1]
#         print("free n="+ str(10**(j+1))+", step=" + str(i)+"gdp=",'{:.3e} ± {:.3e}'.format(mean, std))


def print_data(alg, data):
    for j in range(len(data)):
        for i in range(len(data[j])):
            if np.isnan(data[j][i][0]):
                break
            mean = data[j][i][0]
            std = data[j][i][1]
        print(alg + " n=" + str(10 ** (j + 1)) + ", step=" + str(i) + "gdp=", '{:.3e} ± {:.3e}'.format(mean, std))
# print_data('ga', ga_data)
# print_data('maddpg', maddpg)
print_data('ppo', ppo)
# print_data('bmfac', bmfac)
