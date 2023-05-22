# AI-TaxingPolicy

The optimization of fiscal policies by governments to stimulate economic growth, ensure social equity and stability, and maximize social welfare has been a subject of significant interest. Simultaneously, individuals keenly observe government policies to optimize their own production and saving strategies. 

To simulate this problem, we propose a multi-agent reinforcement learning simulator based on the Bewley-Aiyagari model. Our simulator encompasses various economic activities of governments, households, technology, and financial intermediaries. By integrating reinforcement learning algorithms, it enables the derivation of optimal strategies for governments and individuals while facilitating the study of the relationship between government policies, micro-level household behaviors, and macroeconomic phenomena.


## Installation

```bash
git clone https://github.com/jidiai/AI-TaxingPolicy.git
```

todo : 未来改成 pip install ai_tax



## Requirements

1. Build a Python virtual environment

   ```bash
   conda create -n ai_tax python=3.6
   ```

2. Activate the virtual environment

   ```bash
   conda activate ai_tax
   ```

3. Install the requirements package

   ```bash 
   pip install -r requirements.txt
   ```



## Run

```bash
cd AI-TaxingPolicy
```

1. Free market policy:

   ```bash
   python main.py --device-num 0 --n_households 100 --alg "rule_based"
   ```

2. Independent PPO:

   ```bash
   python main.py --device-num 0 --n_households 100 --alg "ppo"
   ```

3. MAPPO:

   ```bash
   python main.py --device-num 0 --n_households 100 --alg "mappo"
   ```

4. BMFAC:

   ```bash
   python main.py --device-num 0 --n_households 100 --alg "bmfac"
   ```

   "device-num" means GPU index



## Env API

The ai_tax API's API models environments as simple Python `env` classes. Creating environment instances and interacting with them is very simple- here's an example using the "gdp" environment: 
you can change government task in "./cfg/default.yaml".
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



## Citation

If you use ai_tax for published research, please cite:

==todo修改成我们的论文链接== 

```bibtex
@inproceedings{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012},
  organization={IEEE},
  doi={10.1109/IROS.2012.6386109}
}
```

