# TaxAI

The optimization of fiscal policies by governments to stimulate economic growth, ensure social equity and stability, and maximize social welfare has been a subject of significant interest. Simultaneously, individuals keenly observe government policies to optimize their own production and saving strategies. 

To simulate this problem, we propose a multi-agent reinforcement learning simulator based on the Bewley-Aiyagari model. Our simulator encompasses various economic activities of governments, households, technology, and financial intermediaries. By integrating reinforcement learning algorithms, it enables the derivation of optimal strategies for governments and individuals while facilitating the study of the relationship between government policies, micro-level household behaviors, and macroeconomic phenomena.


## Installation

```bash
git clone https://github.com/jidiai/TaxAI.git
```


## Requirements

1. Build a Python virtual environment

   ```bash
   conda create -n TaxAI python=3.6
   ```

2. Activate the virtual environment

   ```bash
   conda activate TaxAI
   ```

3. Install the requirements package

   ```bash 
   pip install -r requirements.txt
   ```



## Run

```bash
cd TaxAI
python main.py --device-num 0 --n_households 1000 --alg "ppo" --task "gdp" --seed 2 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
```

alg = {
        Free market policy: "rule_based",
        Independent PPO: "ppo",
        MADDPG: "maddpg",
        BMFAC: "bmfac" }

task = {max GDP:"gdp", min Gini: "gini", max social welfare: "social_welfare"}

   "device-num" means GPU index



## Env API

The TaxAI API's API models environments as simple Python `env` classes. Creating environment instances and interacting with them is very simple- here's an example using the "gdp" environment:
you can change government task in './cfg/default.yaml'.
```bash
gov_task: "gdp"  # choices: {"gdp", "gini", "social_welfare", "gdp_gini"}
```

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

for _ in range(1000):
    gov_action = env.government.action_space.sample()
    house_action = env.households.action_space.sample()

    action = {env.government.name: gov_action * gov_action_max,  # gov_action & house_action is in (-1,+1)
              env.households.name: house_action * house_action_max}
    next_global_obs, next_private_obs, gov_reward, house_reward, done = env.step(action)
    print("gov reward:", gov_reward, "--- households reward:", house_reward)

if done:
    global_obs, private_obs = env.reset()
env.close()

```

## Acknowledgement
[Reinforcement-learning-algorithms](https://github.com/TianhongDai/reinforcement-learning-algorithms)

[MADDPG](https://github.com/starry-sky6688/MADDPG)