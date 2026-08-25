from pathlib import Path
import sys
import numpy as np

pages = Path(__file__).parent.parent / "pages"
sys.path.append(str(pages))
from onpolicy.envs.drone.maps.map import Map
from onpolicy.envs.drone.pages.drone_sim import load_episode_path, load_target_weapon
from onpolicy.envs.drone.weapons.interfaces.environment_interface import EnvironmentInterface
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
n_task_uavs = cf.getint("env", "n_task_uavs")  # 任务机数量
n_decoy_uavs = cf.getint("env", "n_decoy_uavs")  # 诱饵机数量
n_total_uavs = n_task_uavs + n_decoy_uavs  # 无人机总数量
uav_length = cf.getfloat("env", "uav_length")  # 无人机长度

# 初始化地图
map_data_file = cf.get("map", "map_data_file")  # 地图文件路径
map_resolution = cf.getfloat("map", "map_resolution")  # 地图分辨率
mmap = Map(map_data_file, map_resolution, uav_length)

target, weapon = load_target_weapon()

paths = load_episode_path()

for i, path in enumerate(paths):
    if i == 0:
        EnvironmentInterface.reset(n_total_uavs, weapon,
                                   [paths[0]['uva_state'][0]['velocity']],
                                   [paths[0]['uva_state'][0]['position']], mmap)

    position = [paths[i]['uva_state'][0]['position']]
    velocity = [paths[i]['uva_state'][0]['velocity']]
    game_uav_list = EnvironmentInterface.step(position, velocity)
