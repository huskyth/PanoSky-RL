from pathlib import Path
import sys
import numpy as np
from onpolicy.scripts.train.train_drone import parse_args
from onpolicy.envs.drone.mul_uav_env import MultiUavEnv
from onpolicy.config import get_config
import configparser

temp = Path(__file__).parent.parent
sys.path.append(str(temp))

# 读取配置文件中的参数并保存
config_path = temp / 'config/th_demo.ini'
cf = configparser.ConfigParser()
cf.read(str(config_path), encoding="utf-8")

parser = get_config()
all_args = parse_args(sys.argv[1:], parser)
env = MultiUavEnv(cf=cf, is_debug=True, episode_limit=100, rank=0)

env.reset()
act = []
for _ in range(env.n_total_uavs):
    act.append([0., 0., 0., 0., 1., 0., 0., 0., 0.])

while True:
    env.step(np.array(act))
    if all(env.is_terminal):
        break
