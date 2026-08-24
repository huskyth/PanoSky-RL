import copy
import json
import math
import random
import warnings

import os
import numpy as np
from gym import spaces

from onpolicy.envs.drone.weapons.entries.phalanx.components.track_rader.track_rader_enum import TrackStateEnum
from onpolicy.utils.format_logger import AppLogger, _green_log_str
from onpolicy.envs.drone.weapons.entries.uav.uav_enum import UAVState, AttackState
from onpolicy.envs.drone.maps.map import Map
from onpolicy.utils.util import compute_distance
from onpolicy.envs.drone.uav_meta_info import TrainUAV
from pathlib import Path
from onpolicy.envs.drone.weapons.interfaces.environment_interface import EnvironmentInterface
from onpolicy.utils.math_tool import fly_from_9_selections, angle_2_radian, normalize, cal_threat_level

warnings.filterwarnings('ignore')
logger = AppLogger().get_logger()


def cos_theta(target, position, v):
    to = np.array(target) - np.array(position)
    cos_ = (to * np.array(v)).sum() / (to ** 2).sum() ** 0.5 / (np.array(v) ** 2).sum() ** 0.5
    logger.info(f"cos = {cos_}")
    return cos_


json_path = Path(__file__).parent / "jsons"
if not json_path.exists():
    json_path.mkdir()


# 武器环境已知的环境
class MultiUavEnv:
    # 这边先垂直后水平，有问题再说
    ACTION_SET = [
        (angle_2_radian(10), angle_2_radian(-10)),
        (angle_2_radian(10), angle_2_radian(0)),
        (angle_2_radian(10), angle_2_radian(10)),
        (angle_2_radian(0), angle_2_radian(-10)),
        (angle_2_radian(0), angle_2_radian(0)),
        (angle_2_radian(0), angle_2_radian(10)),
        (angle_2_radian(-10), angle_2_radian(-10)),
        (angle_2_radian(-10), angle_2_radian(0)),
        (angle_2_radian(-10), angle_2_radian(10)),
    ]

    def dump(self, reason):
        if self.is_debug and len(self.episode_data):
            with open(
                    str(json_path / f"{self.mode}模式下第{self.n_episode}个Epoch一共{self._episode_steps}步，【结果】：{reason.split('-')[0]}.json"),
                    'w',
                    encoding="utf-8") as f:
                data = {'target': self.target, 'weapon': self.weapon, 'episode_data': self.episode_data,
                        "reason": reason,
                        "map_max_x": self.map.map_max_x,
                        "map_max_y": self.map.map_max_y,
                        "max_available_height": self.max_available_height,
                        "n_uav": self.n_total_uavs,
                        }

                json.dump(data, f, ensure_ascii=False)

                self.episode_data = []
                import gc
                gc.collect()

    def init_from_config(self, cf):
        self.map_output_dimension = cf.getint("env", "map_output_dimension")
        self.task_success_radius = cf.getfloat("env", "task_success_radius")  # 任务成功半径 m
        self.uav_obs_radius = cf.getfloat("env", "uav_obs_radius")  # 局部观测半径 m
        self.uav_velocity_value = cf.getfloat("env", "uav_velocity_value")  # 飞行速度 m/s
        self.uav_length = cf.getfloat("env", "uav_length")  # 无人机长度
        self.coll_safe_dis = cf.getfloat("env", "coll_safe_dis")  # 碰撞安全距离
        self.dis_target_weapon = cf.getfloat("env", "dis_target_weapon")  # 目标和武器的距离
        self.dis_target_uav_min = cf.getint("env", "dis_target_uav_min")  # 目标和UAV的最小距离
        self.dis_target_uav_max = cf.getint("env", "dis_target_uav_max")  # 目标和UAV的最大距离

        self.uva_init_height = cf.getfloat("env", "uva_init_height")  # 无人机初始高度

        self.max_available_height = cf.getfloat("constraints", "max_available_height")  # 最高可飞行高度
        self.min_available_height = cf.getfloat("constraints", "min_available_height")  # 最低可飞行高度

        # 有关奖励函数的参数
        self.decoy_safe_dis = cf.getfloat("reward", "decoy_safe_dis")  # 引诱的安全距离
        self.uav_coll_penalty = cf.getfloat("reward", "uav_coll_penalty")  # UAV碰撞时的惩罚
        self.step_penalty = cf.getfloat("reward", "step_penalty")  # 执行动作的惩罚
        self.decoy_success_reward = cf.getfloat("reward", "decoy_success_reward")  # 引诱成功的奖励
        self.decoy_dis_reward_index = cf.getfloat("reward", "decoy_dis_reward_index")  # 引诱距离奖励的比例系数
        self.task_success_reward = cf.getfloat("reward", "task_success_reward")  # 任务成功的奖励
        self.task_dis_reward_index = cf.getfloat("reward", "task_dis_reward_index")  # 任务距离奖励的比例系数

        # 计算无人机数量
        self.n_task_uavs = cf.getint("env", "n_task_uavs")  # 任务机数量
        self.n_decoy_uavs = cf.getint("env", "n_decoy_uavs")  # 诱饵机数量
        self.n_total_uavs = self.n_task_uavs + self.n_decoy_uavs  # 无人机总数量

        # 初始化地图
        map_data_file = cf.get("map", "map_data_file")  # 地图文件路径
        map_resolution = cf.getfloat("map", "map_resolution")  # 地图分辨率
        self.map = Map(map_data_file, map_resolution, self.uav_length)

        assert self.task_success_radius > self.uav_velocity_value

    def init_space(self):
        # 定义状态空间：线性数据 + 图像

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        obs_dim = None
        for agent_id in range(self.n_total_uavs):
            self.action_space.append(spaces.Discrete(9))
            obs_dim = self.get_observation_size_of_a_uav()
            if self.is_share:
                observation_space = spaces.Dict({
                    # 1) 线性数据：一维连续向量（Box）
                    "linear": spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32),

                    # 2) 图像数据：高/宽/通道（最常用格式 HWC）
                    # "image": spaces.Box(
                    #     low=0,  # 像素 0~255
                    #     high=255,
                    #     shape=(1, 32, 32),  # 高84 × 宽84 × 3通道(RGB)
                    #     dtype=np.uint8
                    # )
                })
            else:
                observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
            self.observation_space.append(observation_space)

        if self.is_share:
            cent_obs_define = spaces.Dict({
                # 1) 线性数据：一维连续向量（Box）
                "linear": spaces.Box(low=-np.inf, high=+np.inf, shape=(self.n_total_uavs * obs_dim,), dtype=np.float32),

                # 2) 图像数据：高/宽/通道（最常用格式 HWC）
                # "image": spaces.Box(
                #     low=0,  # 像素 0~255
                #     high=255,
                #     shape=(self.n_total_uavs, 32, 32),  # 高84 × 宽84 × 3通道(RGB)
                #     dtype=np.uint8
                # )
            })
        else:
            cent_obs_define = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.n_total_uavs * obs_dim,),
                                         dtype=np.float32)
        self.share_observation_space = [
            cent_obs_define
            for _ in range(self.n_total_uavs)]

    def __init__(self, rank, mode="train", cf=None, episode_limit=500, is_debug=False, is_share=True):
        self.rank = rank
        # train为训练模式，test为测试模式
        self.mode = mode
        self.is_debug = is_debug
        self.is_use_weapon = True
        self.is_share = is_share
        self.right_vector = None

        # 初始化回合数
        self.n_episode = 0
        self._episode_steps = 0
        self.max_episode_steps = episode_limit  # 任务的最大回合数
        logger.info(f"最大步数为 {episode_limit}")
        # 从游戏获取参数并保存
        self.radar_detect_radius = EnvironmentInterface.get_rader_detect_radius()  # 雷达检测半径 m
        self.weapon_fire_radius = EnvironmentInterface.get_fire_distance()  # 武器开火半径 m

        # 初始化无人机集群
        self.raw_uavs = []
        self.episode_data = []
        # 初始化目标和武器
        self.target = [0, 0, 0]
        self.weapon = [0, 0, 0]

        self.init_from_config(cf)
        self.init_space()

        self.reward = None
        self.is_terminal = [False for _ in range(self.n_total_uavs)]

        logger.info(
            f"PID-{os.getpid()}, 【{'训练' if self.mode == 'train' else '评估'}】环境（武器位置已知）初始化完成，"
            f"任务机数量为{self.n_task_uavs}，诱饵机数量为{self.n_decoy_uavs}，是否使用武器【{self.is_use_weapon}】")

    def _init_xyz(self):
        available = False

        # 目标点坐标
        target_x, target_y, target_z = self.target[0], self.target[1], self.target[2]
        uav_x, uav_y = 0, 0
        while not available:
            # 1. 固定高度
            uav_z = self.uva_init_height

            # 2. 安全高度校验（不满足直接跳过，重新生成）
            min_safe_z = float(self.map.search_nh(uav_x, uav_y) + 2 * self.coll_safe_dis)
            assert uav_z > min_safe_z

            # 3. 计算垂直高度差
            dz = uav_z - target_z

            r = random.uniform(self.dis_target_uav_min, self.dis_target_uav_max)
            # 4. 勾股定理计算 必须的水平距离（保证3D距离严格固定）
            # 水平距离² + 垂直距离² = 固定3D距离²
            horizontal_sq = r ** 2 - dz ** 2
            # 防止高度差超过3D距离导致无实数解
            if horizontal_sq < 0:
                continue
            horizontal_r = math.sqrt(horizontal_sq)

            # 5. 随机水平角度（0~2π），保证在目标点周围均匀生成
            angle = random.uniform(0, 2 * math.pi)

            # 6. 计算固定水平距离的坐标（核心！）
            uav_x = target_x + horizontal_r * math.cos(angle)
            uav_y = target_y + horizontal_r * math.sin(angle)

            # 7. 判断位置是否可用
            available = self.judge_random_position_available(uav_x, uav_y, uav_z)

        return uav_x, uav_y, uav_z

    def reset(self):
        """
        每个UAV的状态包含 [x, y, z, v_x, v_y, v_z, decoy_flag, attacked_state, status]
        decoy_flag表示无人机是否为诱饵机，诱饵机为1，任务机为2
        attacked_state，不被攻击为0，被攻击为1，被摧毁为2
        status表示无人机状态，存活为"Alive"，死亡为"UAV_collision/Ground_collision/Destroyed"

        训练环境随机生成武器、目标和无人机集群的初始位置
        测试环境需要中通过传参控制武器、目标和无人机集群的初始位置
        """
        self.target = [0, 0, 0]
        self.weapon = [0, 0, 0]
        self.raw_uavs = []
        self.episode_data = []  # 存储历史的数据
        self._episode_steps = 0
        self.right_vector = [[1, 0, 0] for i in range(self.n_total_uavs)]
        self.reward = None
        self.is_terminal = [False for _ in range(self.n_total_uavs)]

        # 重置目标位置
        # 为了保证目标和武器都在地图内，预留dis_target_weapon的空间

        self.target[0] = float(random.uniform(self.map.map_min_x + self.dis_target_weapon,
                                              self.map.map_max_x - self.dis_target_weapon))
        self.target[1] = float(random.uniform(self.map.map_min_y + self.dis_target_weapon,
                                              self.map.map_max_y - self.dis_target_weapon))

        self.target[2] = float(self.map.search_nh(self.target[0], self.target[1]) + self.coll_safe_dis)
        # 重置武器位置 theta 为 目标和武器连线与X轴正向夹角 范围为 [0，2pai）
        theta_w = 2 * math.pi * random.uniform(0, 1)
        self.weapon[0] = float(self.target[0] + self.dis_target_weapon * math.cos(theta_w))
        self.weapon[1] = float(self.target[1] + self.dis_target_weapon * math.sin(theta_w))
        self.weapon[2] = float(self.map.search_nh(self.weapon[0], self.weapon[1]) + self.coll_safe_dis)
        # 你的循环代码（已修改为 3D 真实距离）
        for uav in range(self.n_total_uavs):
            # 初始化速度（原有逻辑）
            uav_x, uav_y, uav_z = self._init_xyz()
            init_vel = self._init_toward_velocity([uav_x, uav_y, uav_z], self.target)
            temp_uav = TrainUAV(uav_x, uav_y, uav_z, *init_vel, AttackState.SAFE, UAVState.ALIVE)
            self.raw_uavs.append(temp_uav)

        # 把无人机总数量、UAV初始位置、武器初始位置传给游戏
        if self.is_use_weapon:
            EnvironmentInterface.reset(self.n_total_uavs, self.weapon,
                                       [self.raw_uavs[i].velocity for i in range(self.n_total_uavs)],
                                       [self.raw_uavs[i].position for i in range(self.n_total_uavs)], self.map)
        which_idx = self._get_game_target_idx()
        data_save = {"uva_state": [x.to_dict() for x in self.raw_uavs],
                     "uva_actions": [-1 for _ in range(self.n_total_uavs)],
                     "_episode_steps": self._episode_steps, "reward": self.reward,
                     "right_vec": self.right_vector,
                     "c_target_id": id(self.raw_uavs[which_idx]) if which_idx is not None else "None"}
        self.episode_data.append(data_save)

        # 获取UAV集群观测并返回

        return self.get_state_of_all_uav()

    def _init_toward_velocity(self, uav, target):
        init_vel = [0, 0, 0]
        three_dim_dis = compute_distance(uav, target)
        init_vel[2] = (target[2] - uav[2]) * self.uav_velocity_value / three_dim_dis
        sin_beta = (target[2] - uav[2]) / three_dim_dis
        cos_beta = (1 - sin_beta ** 2) ** 0.5
        two_dim_dis = ((target[0] - uav[0]) ** 2 + (target[1] - uav[1]) ** 2) ** 0.5
        init_vel[0] = (target[0] - uav[0]) * self.uav_velocity_value * cos_beta / two_dim_dis
        init_vel[1] = (target[1] - uav[1]) * self.uav_velocity_value * cos_beta / two_dim_dis
        return init_vel

    # 判断随机生成的UAV位置是否可用
    def judge_random_position_available(self, uav_x, uav_y, uav_z):
        if uav_x < self.map.map_min_x or uav_x > self.map.map_max_x:
            return False
        if uav_y < self.map.map_min_y or uav_y > self.map.map_max_y:
            return False
        available = True
        if len(self.raw_uavs) > 0:
            # 循环当前无人机列表判断是否位置是否会相互碰撞
            for i in range(len(self.raw_uavs)):
                dis = compute_distance([uav_x, uav_y, uav_z], self.raw_uavs[i].position)
                # 如果距离会导致与当前任何一架无人机碰撞则返回False重新生成
                if dis < self.uav_length + self.coll_safe_dis:
                    available = False
        return available

    def get_state_of_all_uav(self):
        return [self.get_observation_of_a_uav(i) for i in range(self.n_total_uavs)]

    def get_observation_of_a_uav(self, uav_id):
        """
            uav_id前一半为牺牲机，后一半为攻击机
        """

        def get_status_value(status):
            if status == UAVState.ALIVE:
                return 0
            if status == UAVState.DESTROYED:
                return 1
            if status == UAVState.COLLISION:
                return 0

        assert self.is_use_weapon is True
        normal_ = np.array([self.map.map_max_x, self.map.map_max_y, self.max_available_height])
        # 友军的位置和速度、受攻击状态
        # TODO://待检查，友军和武器有关，观测半径可以后续再加

        # position = self.raw_uavs[uav_id].position
        # map_obs = self.map.generate_img(position[0], position[1])[0][0]
        team = []
        target_idx = self._get_game_target_idx()
        target_idx = -1 if target_idx is None else target_idx

        def set_uav_state(idx):
            nonlocal team
            tp_uav = self.raw_uavs[idx]
            position = tp_uav.position
            velocity = tp_uav.velocity
            right_vector = self.right_vector[idx]
            w_state = EnvironmentInterface.get_weapon_state()
            dis = compute_distance(self.weapon, position)
            c_tar = target_idx
            player = int(idx == 0)
            status = get_status_value(tp_uav.status)
            if tp_uav.status != UAVState.ALIVE:
                position, velocity, right_vector = [0, 0, 0], [0, 0, 0], [0, 0, 0]
                w_state, dis, c_tar, player, status = 0, 0, 0, 0, 0
            team += position
            team += velocity
            team += right_vector
            team += [w_state]
            team += [dis]
            team += [c_tar]
            team += [player]
            team += [status]

        set_uav_state(uav_id)

        for i in range(self.n_total_uavs):
            if i == uav_id:
                continue
            # 本机位置和速度
            set_uav_state(i)

        weapon = self.weapon
        target = self.target
        team += weapon
        team += target
        if self.is_share:
            return team, [None]
        else:
            return team

    def _get_game_target_idx(self):
        c_target = EnvironmentInterface.try_get_current_target()
        if c_target is None:
            return None
        for i in range(self.n_total_uavs):
            if c_target is EnvironmentInterface.get_uav_list()[i]:
                return i
        assert False

    def step(self, action):
        # 输入动作先转换为两个离散动作
        actions = []
        # 对于每个智能体
        for idx in range(self.n_total_uavs):
            action_index = int(np.where(action[idx] == 1)[0])
            actions.append([action_index])

        # 获取执行动作前的UAV集群观测
        last_state = copy.deepcopy(self.raw_uavs)
        # 计算每架UAV执行动作后的速度、位置、碰撞情况以及与武器之间是否存在山体遮挡
        for idx in range(self.n_total_uavs):
            if self.raw_uavs[idx].status != UAVState.ALIVE:
                continue
            actual_action = actions[idx][0]
            radian_action = self.ACTION_SET[actual_action]

            # 计算执行动作之后的速度和位置
            temp = self.raw_uavs[idx]
            velocity = temp.velocity
            position = temp.position
            right = self.right_vector[idx]
            next_position, after_rotate_velocity_direction_in_parent, after_rotate_horizontal_right_vector_in_parent \
                = fly_from_9_selections(*radian_action, velocity, right, position)
            actual_velocity = (np.array(after_rotate_velocity_direction_in_parent) * self.uav_velocity_value).tolist()
            self.right_vector[idx] = after_rotate_horizontal_right_vector_in_parent
            # 判断是否会发生碰撞
            # self.judge_uav_collision_and_set(idx, *next_position)
            # 执行动作改变UAV状态
            self.raw_uavs[idx].set_position(*next_position)
            self.raw_uavs[idx].set_velocity(*actual_velocity)
        self._episode_steps += 1  # 当前的回合数增加上动作执行的回合数。

        # 把位置传给游戏，然后获取执行动作后的受击状态并保存
        target_idx_before = None
        # TODO://待检查，先注释，训练完成了再解开
        if self.is_use_weapon:
            lat_stat = EnvironmentInterface.get_weapon_state()
            position = [self.raw_uavs[u].position for u in range(self.n_total_uavs)]
            velocity = [self.raw_uavs[u].velocity for u in range(self.n_total_uavs)]
            target_idx_before = self._get_game_target_idx()

            game_uav_list = EnvironmentInterface.step(position, velocity)
            for u in range(self.n_total_uavs):
                if game_uav_list[u] is None:
                    self.raw_uavs[u].status = UAVState.DESTROYED

        # 计算奖励值和终止符号
        self.set_reward(last_state, actions, target_idx_before, lat_stat)

        ret_reward = [[x] for x in self.reward]
        return self.get_state_of_all_uav(), ret_reward, self.is_terminal, {i: {'individual_reward': self.reward[i]} for
                                                                           i in range(self.n_total_uavs)}

    def append_data(self, action):
        which_idx = self._get_game_target_idx()
        data_save = {"uva_state": [x.to_dict() for x in self.raw_uavs], "uva_actions": action,
                     "right_vec": self.right_vector,
                     "_episode_steps": self._episode_steps, "reward": self.reward,
                     "c_target_id": id(self.raw_uavs[which_idx]) if which_idx is not None else "None",
                     'r_msg': self.r_msg, 'degree': self.degree
                     }
        self.episode_data.append(data_save)

    def judge_uav_collision_and_set(self, current_uav_idx, next_x, nex_y, next_z):
        cur_uav = self.raw_uavs[current_uav_idx]
        p_x, p_y, p_z = cur_uav.position
        # 判断是否与地面发生碰撞
        if self.map.judge_mountain(p_x, p_y, p_z, next_x, nex_y, next_z, self.coll_safe_dis, 'ground'):
            # 如果路径中碰撞也算
            cur_uav.status = UAVState.COLLISION

        # 判断是否超界
        if (
                next_x < self.map.map_min_x or next_x > self.map.map_max_x or nex_y < self.map.map_min_y or nex_y > self.map.map_max_y
                or next_z < self.min_available_height or next_z > self.max_available_height):
            cur_uav.status = UAVState.COLLISION

    def set_reward(self, last_p, action, last_target, lat_stat):
        current_p = self.raw_uavs

        if self.is_use_weapon:
            self.reward = [-0.1 for _ in range(self.n_total_uavs)]
            self.r_msg = ['' for _ in range(self.n_total_uavs)]
            self.degree = [None for _ in range(self.n_total_uavs)]
            is_reach = False
            is_finish = False
            is_col = False
            dis = [None for _ in range(self.n_total_uavs)]
            target_idx = self._get_game_target_idx()
            for idx in range(self.n_total_uavs):
                deg = cal_threat_level(self.weapon, self.uav_velocity_value,
                                       current_p[idx].velocity, current_p[idx].position)
                self.degree[idx] = deg

                dis[idx] = compute_distance(current_p[idx].position, self.target)
                if dis[idx] <= self.task_success_radius and idx == 1:
                    self.reward[idx] += 200
                    is_reach = True
                    self.r_msg[idx] = f'{idx}到了，'

                if current_p[idx].status == UAVState.DESTROYED:
                    self.reward[idx] -= 100
                    self.is_terminal[idx] = True
                    self.r_msg[idx] += f'{idx}被摧毁了，'

            # if current_p[i].status == UAVState.COLLISION:
            #     self.reward[i] -= 6
            #     self.is_terminal[i] = True
            #     self.r_msg[i] += f'{i}撞地了-'
            # 0诱饵机 1突防机
            if last_target is None:
                if target_idx is None:
                    self.reward[0] += -1
                    self.r_msg[0] += '要去勾引，'
                elif target_idx == 0:
                    self.reward[0] += 1
                    self.r_msg[0] += '勾引到了，'
                else:
                    self.reward[0] -= 20
                    self.r_msg[0] += '1被瞄了，'
            elif last_target == 0:
                if target_idx is None:
                    if current_p[0].status == UAVState.ALIVE and lat_stat == TrackStateEnum.FIRE.value:
                        self.reward[0] += 3
                        self.r_msg[0] += '0勾引到了并且解除了，'
                        gre = _green_log_str('0勾引到了并且解除了')
                        logger.info(f"PID-{os.getpid()}, mode-{self.mode}, episode-{self.n_episode} {gre}")
                else:
                    self.reward[0] += -1
                    self.r_msg[0] += '0被瞄中需要尽快解除，'
            else:
                if target_idx is None:
                    if current_p[1].status == UAVState.ALIVE:
                        self.reward[1] += 4
                        self.r_msg[1] += '1勾引到了并且解除了，'
                else:
                    self.reward[1] += -10
                    self.r_msg[1] += '1被瞄中需要尽快解除，'

                # c_dis = compute_distance(current_p[i].position, self.target)
                # if c_dis <= self.task_success_radius:
                #     self.reward[i] += 10
                #     is_finish = True
                #     self.r_msg[i] += f'{i}到达目的地，'

                # if abs(c_dis - 1500) < self.uav_velocity_value + 1 and i == 1:
                #     self.reward[i] += 1
                #     self.r_msg[i] += f'{i}到达1500附近，'

                # x, y, z = current_p[i].position

                # l_dis = compute_distance(last_p[i].position, self.target)
                # if l_dis >= c_dis:
                #     self.reward[i] += 0.1
                #     self.r_msg[i] += '你近了，'
                # else:
                #     self.reward[i] += -0.1
                #     self.r_msg[i] += '你远了，'

                # hm = self.map.search_nh(x, y)
                # det = z - hm
                # info = f'z={z}，hm={hm}，'
                # self.r_msg[i] = info + self.r_msg[i]
                # if det > 200:
                #     self.reward[i] -= 1
                #     self.r_msg[i] += '飞太高了，'
                # elif 200 >= det > 100:
                #     self.reward[i] += 1
                #     self.r_msg[i] += '贴地飞行，'
                # else:
                #     self.reward[i] -= 1
                #     self.r_msg[i] += '飞太低，'

                # if self.map.judge_mountain(*self.weapon, *current_p[i].position, 0, 'block'):
                #     # TODO://不能在地下
                #     pass
                # self.reward[i] += 1.5
                # self.r_msg[i] += '被山遮挡，'

            # if is_finish:
            #     self.append_data(action)
            #     msg = "到达目的地"
            #     self.dump(msg)
            #     self.is_terminal = [True for _ in range(self.n_total_uavs)]
            #     green_str = _green_log_str(f"[terminated]：{msg}")
            #     logger.info(f"PID-{os.getpid()}, mode-{self.mode}, episode-{self.n_episode} {green_str}")
            #     self.n_episode = self.n_episode + 1
            #     return

            if is_reach:
                self.append_data(action)
                self.is_terminal = [True for _ in range(self.n_total_uavs)]
                msg = "任务完成！"
                self.dump(msg)
                green_str = _green_log_str(msg)
                logger.info(f"PID-{os.getpid()}, mode-{self.mode}, episode-{self.n_episode} {green_str}")
                self.n_episode = self.n_episode + 1
                return

            if all(self.is_terminal):
                self.append_data(action)
                msg = "都被击毁！"
                self.dump(msg)
                green_str = msg
                logger.info(f"PID-{os.getpid()}, mode-{self.mode}, episode-{self.n_episode} {green_str}")
                self.n_episode = self.n_episode + 1
                return

            if self._episode_steps >= self.max_episode_steps:
                for i in range(self.n_total_uavs):
                    self.r_msg[i] += '步子到了。'
                self.n_episode = self.n_episode + 1
                self.is_terminal = [True for _ in range(self.n_total_uavs)]
                logger.info(
                    f"PID-{os.getpid()}, mode-{self.mode}, "
                    f"episode-{self.n_episode}\033[31m[terminated]：超出最大限制 {self.r_msg}\033[0m")
                self.append_data(action)
                self.dump("超出最大限制")
                return

            self.append_data(action)

    def compute_init_velocity(self):
        # 保持设定速率不变
        speed = self.uav_velocity_value
        # 生成三维随机方向单位向量
        # 球坐标系随机方位角、俯仰角
        yaw = random.uniform(0, 2 * math.pi)  # 水平0~360度
        pitch = random.uniform(-math.pi / 2, math.pi / 2)  # 俯仰-90~90度

        # 球面坐标转笛卡尔单位方向向量
        dx = math.cos(pitch) * math.cos(yaw)
        dy = math.cos(pitch) * math.sin(yaw)
        dz = math.sin(pitch)

        # 合成随机方向、固定速率的速度
        init_vel = [
            dx * speed,
            dy * speed,
            dz * speed
        ]
        return init_vel

    def seed(self, seed_num=None):
        pass

    def get_observation_size_of_a_uav(self):
        """ Returns the shape of the observation
        注意：地形额外加
        （位置、速度、受攻击状态、是否是诱饵机）--按照次序--N * 8
        8: 自身-位置、速度、受攻击状态、是否是诱饵机
        3: 目标坐标
        3: 武器坐标
        3: 考虑到右向量和飞行有关，所以要加进去
        (N * 8 + 8 + 3 + 3)
        """
        if self.is_use_weapon:
            return self.n_total_uavs * 14 + 6
        else:
            return 3 + 3 + 3 + 3 + 3

    def close(self):
        pass
