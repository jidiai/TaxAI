# TaxAI: A Dynamic Economic Simulator and Benchmark for Multi-Agent Reinforcement Learning

<div style="text-align:center">
  <img src="./img/new_model_dynamics.png" alt="示例图片" >
  <figcaption style="text-align:center;"></figcaption>
</div>


The optimization of fiscal policies by governments to stimulate economic growth, ensure social equity and stability, and maximize social welfare has been a subject of significant interest. Simultaneously, individuals keenly observe government policies to optimize their own production and saving strategies. 

To simulate this problem, we propose a multi-agent reinforcement learning simulator based on the Bewley-Aiyagari model. Our simulator encompasses various economic activities of governments, households, technology, and financial intermediaries. By integrating reinforcement learning algorithms, it enables the derivation of optimal strategies for governments and individuals while facilitating the study of the relationship between government policies, micro-level household behaviors, and macroeconomic phenomena.

### A comparison of MARL simulators for optimal taxation problems

| Simulator             | AI Economist | RBC Model  | **TaxAI** (ours) |
|-------------------------|--------------|--------------|---------------------|
| Households' Number    | 10           | 100        | 10000               |
| Tax Schedule          | Non-linear   | Linear     | Non-linear          |
| Tax Type              | Income       | Income     | Income & Wealth & Consumption  |
| Social Roles' Types   | 2            | 3          | 4                   |
| Saving Strategy       | &#x2716;    | &#x2714;| &#x2714;         |
| Heterogenous Agent    | &#x2714;     | &#x2714;| &#x2714;         |
| Real-data Calibration | &#x2716;    | &#x2716;  | &#x2714;         |
| Open source           | &#x2714;  | &#x2716;  | &#x2714;         |
| MARL Benchmark        | &#x2716;    | &#x2716;  | &#x2714;         |

Our paper: 

TaxAI: A Dynamic Economic Simulator and Benchmark for Multi-Agent Reinforcement Learning 
[https://arxiv.org/abs/2309.16307](https://arxiv.org/abs/2309.16307)

## Install

You can use any tool to manage your python environment. Here, we use conda as an example.

1. Install conda/miniconda.

2. Build a Python virtual environment.
```bash
conda create -n TaxAI python=3.6
```

3. Activate the virtual environment

```bash
conda activate TaxAI
```

4. Clone the repository and install the required dependencies
```bash 
git clone https://github.com/jidiai/TaxAI.git
cd TaxAI
pip install -r requirements.txt
```

## Execution
After installation, run an example experiment by executing the following command from the home folder:
```bash
python run_a_game.py
```
or run python code as follows:

```python
from env.env_core import economic_society
from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(f'./cfg/default.yaml')  # get environment parameters
env = economic_society(yaml_cfg.Environment)

# get the action max
gov_action_max = env.government.action_space.high[0]
house_action_max = env.households.action_space.high[0]

# global obs is observed by gov & households; Private obs are observed separately by each household.
global_obs, private_obs = env.reset()

for _ in range(100):
    gov_action = env.government.action_space.sample()
    house_action = env.households.action_space.sample()

    action = {env.government.name: gov_action * gov_action_max,  # gov_action & house_action is in (-1,+1)
              env.households.name: house_action * house_action_max}
    next_global_obs, next_private_obs, gov_reward, house_reward, done = env.step(action)
    print("gov reward:", gov_reward, "\nhouseholds reward:", house_reward)

    if done:
        global_obs, private_obs = env.reset()
env.close()
```
If users obtain a similar output as follows, it indicates a successful installation of the TaxAI simulator.
```text
gov reward: 0.27574195559515985 
households reward: [[-14.16824042]
 [ 13.33342979]
 [  7.38537561]
 [ 10.8789686 ]
 [-16.69487928]
 [ 10.96102371]
 [-27.29789107]
 [ 11.32268419]
 [-24.14579232]
 [  9.87050757]]
gov reward: -0.028681460944447557 
households reward: [[  7.08354761]
 [ -7.66086619]
 [  4.4566605 ]
 [-19.19874515]
 [  5.28689801]
 [ 10.49161175]
 [  8.16525891]
 [  7.82208646]
 [ -2.99427493]
 [ -1.13584677]]
 ......
```

## Algorithms Benchmark
We support traditional economic methods and multiple MARL algorithms on benchmark scenarios.


### Scenarios
We design 4 tasks in TaxAI, and users can design different weights when optimizing multiple tasks.
```bash
gov_task: "gdp"  # choices: {"gdp", "gini", "social_welfare", "gdp_gini"}
```
- **Maximizing GDP Growth Rate**: The economic growth can be measured by Gross Domestic Product (GDP). Without considering imports and exports in an open economy, GDP is equal to the output $Y_t$ in our model. 
Based on reality, we set the government's objective to maximize the GDP growth rate.

- **Minimizing Social Inequality**: Social equality and stability build the foundation for all social activities. Social inequality is usually measured by the Gini coefficient of wealth distribution $\mathcal{W}_t$ and income distribution $\mathcal{I}_t$. The Gini coefficient is calculated by the ratio of the area between the Lorenz curve and the perfect equality line, divided by the total area under the perfect equality line(shown in figure~\ref{fig:markov game}). The Gini coefficient ranges between 0 (perfect equality) and 1 (perfect inequality).

- **Maximizing Social Welfare**: Social welfare is an important indicator to present the happiness of the population, which is computed by the sum of all households' lifetime utility.

- **Optimizing Multiple Tasks**: If the government aims to simultaneously optimize multiple objectives, we weigh and sum up multiple objectives. The weights $\omega_1$, $\omega_2$ indicate the relative importance of gini and welfare objectives.

### Supported algorithms

(1) Traditional Economic Methods: 
- Free Market Policy
- Genetic Algorithm (GA)

(2) Independent Learning: 
- Independent PPO

(3) Centralized Training Distributed Execution (CTDE): 
- MADDPG
- MAPPO

(4) Heterogeneous-Agent Reinforcement Learning (HARL): 
- HAPPO
- HATRPO
- HAA2C

(5) Mean Field Multi-Agent Reinforcement Learning (MF-MARL): 
- Bi-level Mean Field Actor-Critic (BMFAC)


### Train agents

1. Train free-market agents.
```bash
cd TaxAI
python main.py --n_households 10 --alg "rule_based" --task "gdp" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --n_households 10 --alg "rule_based" --task "gini" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --n_households 10 --alg "rule_based" --task "social_welfare" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --n_households 10 --alg "rule_based" --task "gdp_gini" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
```

2. Train IPPO agents.
```bash
cd TaxAI
python main.py --n_households 10 --alg "ppo" --task "gdp" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --n_households 10 --alg "ppo" --task "gini" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --n_households 10 --alg "ppo" --task "social_welfare" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --n_households 10 --alg "ppo" --task "gdp_gini" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
```

3. Train MADDPG agents.
```bash
cd TaxAI
python main.py --n_households 10 --alg "maddpg" --task "gdp" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --n_households 10 --alg "maddpg" --task "gini" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --n_households 10 --alg "maddpg" --task "social_welfare" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --n_households 10 --alg "maddpg" --task "gdp_gini" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
```

4. Train BMFAC agents.
```bash
cd TaxAI
python main.py --n_households 10 --alg "bmfac" --task "gdp" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --n_households 10 --alg "bmfac" --task "gini" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --n_households 10 --alg "bmfac" --task "social_welfare" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
python main.py --n_households 10 --alg "bmfac" --task "gdp_gini" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
```


5. Train HARL/MAPPO agents.
   (to add)



## Markov Game

<div style="text-align:center">
  <img src="./img/markov games flow.jpg" alt="示例图片" width=80%>
  <figcaption style="text-align:center;"></figcaption>
</div>

The Markov game between the government and household agents. In the center of the figure, we display the Lorenz curves of households' wealth distribution.  The global observation consists of the average assets $\bar{a}_t$, income $\bar{i}_t$, and productivity level $\bar{e}_t$ of the 50\% poorest households and 10\% richest households, along with the wage rate $W_t$. For the government agent, it observes the global observation and takes tax and spending actions $\{\tau_t, \xi_t, \tau_{a,t}, \xi_{a,t}, r^G_t\}$ through the actor network. For household agents, they observe both global and private observation, including personal assets $\{a^i_t\}$ and productivity level $\{e^i_t\}$, and generate savings and workings actions $\{p^i_t, h^i_t\}$ through the actor network. The actor network structure in the figure is just an example.



## Experiment Results

### 1. Optimal Solution in Dynamic Game

<div style="text-align:center">
  <img src="./img/results.jpg" alt="示例图片" width=100%>
  <figcaption style="text-align:center;"></figcaption>
</div>

### 2. Households Dynamic Responses

<div style="text-align:center">
  <img src="./img/tax_action.jpg" alt="示例图片" width=100%>
  <figcaption style="text-align:center;"></figcaption>
</div>

### 3. Scalability of Environment

<div style="text-align:center">
  <img src="./img/scalability.png" alt="示例图片" width=80%>
  <figcaption style="text-align:center;"></figcaption>
</div>

### 4. AI-based Policy Analysis

<div style="text-align:center">
  <img src="./img/maddpg_indicators.jpg" alt="示例图片" width=100%>
  <figcaption style="text-align:center;"></figcaption>
</div>



### 5. Training Curves

<div style="text-align:center">
  <img src="./img/training_curves.jpg" alt="示例图片" width=100%>
  <figcaption style="text-align:center;"></figcaption>
</div>

## Acknowledgement

[Reinforcement-learning-algorithms](https://github.com/TianhongDai/reinforcement-learning-algorithms)

[MADDPG](https://github.com/starry-sky6688/MADDPG)

[HARL](https://github.com/PKU-MARL/HARL)

## Contact
If you have any questions about this repo, feel free to leave an issue. 
You can also contact current maintainers Qirui Mi by email miqirui2021@ia.ac.cn.