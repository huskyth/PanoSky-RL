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


json_path = Path(__file__).parent / "jsons"
if not json_path.exists():
    json_path.mkdir()


class MultiUavEnv:
    # 九方向动作：水平俯仰各 ±10°
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
        self.task_success_radius = cf.getfloat("env", "task_success_radius")
        self.uav_obs_radius = cf.getfloat("env", "uav_obs_radius")
        self.uav_velocity_value = cf.getfloat("env", "uav_velocity_value")
        self.uav_length = cf.getfloat("env", "uav_length")
        self.coll_safe_dis = cf.getfloat("env", "coll_safe_dis")
        self.dis_target_weapon = cf.getfloat("env", "dis_target_weapon")
        self.dis_target_uav_min = cf.getint("env", "dis_target_uav_min")
        self.dis_target_uav_max = cf.getint("env", "dis_target_uav_max")
        self.uva_init_height = cf.getfloat("env", "uva_init_height")
        self.max_available_height = cf.getfloat("constraints", "max_available_height")
        self.min_available_height = cf.getfloat("constraints", "min_available_height")

        # 奖励参数
        self.decoy_safe_dis = cf.getfloat("reward", "decoy_safe_dis")
        self.uav_coll_penalty = cf.getfloat("reward", "uav_coll_penalty")
        self.step_penalty = cf.getfloat("reward", "step_penalty")
        self.decoy_success_reward = cf.getfloat("reward", "decoy_success_reward")
        self.decoy_dis_reward_index = cf.getfloat("reward", "decoy_dis_reward_index")
        self.task_success_reward = cf.getfloat("reward", "task_success_reward")
        self.task_dis_reward_index = cf.getfloat("reward", "task_dis_reward_index")

        self.n_task_uavs = cf.getint("env", "n_task_uavs")
        self.n_decoy_uavs = cf.getint("env", "n_decoy_uavs")
        self.n_total_uavs = self.n_task_uavs + self.n_decoy_uavs

        map_data_file = cf.get("map", "map_data_file")
        map_resolution = cf.getfloat("map", "map_resolution")
        self.map = Map(map_data_file, map_resolution, self.uav_length)

        assert self.task_success_radius > self.uav_velocity_value

        # ===== 新增：课程学习参数 =====
        self.curriculum_stage = 0
        self.curriculum_thresholds = [1.0, 0.5, 0.3]  # 成功率阈值
        self.curriculum_stage_rewards = [0.0, 5.0, 10.0]  # 阶段奖励加成
        self.stage_success_history = []
        self.current_stage_success_count = 0
        self.current_stage_episodes = 0

    def init_space(self):
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        obs_dim = None
        for agent_id in range(self.n_total_uavs):
            self.action_space.append(spaces.Discrete(9))
            obs_dim = self.get_observation_size_of_a_uav()
            if self.is_share:
                observation_space = spaces.Dict({
                    "linear": spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32),
                })
            else:
                observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
            self.observation_space.append(observation_space)

        if self.is_share:
            cent_obs_define = spaces.Dict({
                "linear": spaces.Box(low=-np.inf, high=+np.inf, shape=(self.n_total_uavs * obs_dim,), dtype=np.float32),
            })
        else:
            cent_obs_define = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.n_total_uavs * obs_dim,),
                                         dtype=np.float32)
        self.share_observation_space = [
            cent_obs_define
            for _ in range(self.n_total_uavs)]

    def __init__(self, rank, mode="train", cf=None, episode_limit=500, is_debug=False, is_share=True):
        self.rank = rank
        self.mode = mode
        self.is_debug = is_debug
        self.is_use_weapon = True
        self.is_share = is_share
        self.right_vector = None

        self.n_episode = 0
        self._episode_steps = 0
        self.max_episode_steps = episode_limit
        logger.info(f"最大步数为 {episode_limit}")

        self.radar_detect_radius = EnvironmentInterface.get_rader_detect_radius()
        self.weapon_fire_radius = EnvironmentInterface.get_fire_distance()

        self.raw_uavs = []
        self.episode_data = []
        self.target = [0, 0, 0]
        self.weapon = [0, 0, 0]

        self.init_from_config(cf)
        self.init_space()

        self.reward = None
        self.is_terminal = [False for _ in range(self.n_total_uavs)]

        # ===== 新增：课程学习统计 =====
        self.curriculum_episode_count = 0
        self.curriculum_success_count = 0

        logger.info(
            f"PID-{os.getpid()}, 【{'训练' if self.mode == 'train' else '评估'}】环境（武器位置已知）初始化完成，"
            f"任务机数量为{self.n_task_uavs}，诱饵机数量为{self.n_decoy_uavs}，是否使用武器【{self.is_use_weapon}】")

    def _init_xyz(self):
        available = False
        target_x, target_y, target_z = self.target[0], self.target[1], self.target[2]
        uav_x, uav_y = 0, 0
        while not available:
            uav_z = self.uva_init_height
            min_safe_z = float(self.map.search_nh(uav_x, uav_y) + 2 * self.coll_safe_dis)
            assert uav_z > min_safe_z
            dz = uav_z - target_z
            r = random.uniform(self.dis_target_uav_min, self.dis_target_uav_max)
            horizontal_sq = r ** 2 - dz ** 2
            if horizontal_sq < 0:
                continue
            horizontal_r = math.sqrt(horizontal_sq)
            angle = random.uniform(0, 2 * math.pi)
            uav_x = target_x + horizontal_r * math.cos(angle)
            uav_y = target_y + horizontal_r * math.sin(angle)
            available = self.judge_random_position_available(uav_x, uav_y, uav_z)
        return uav_x, uav_y, uav_z

    # ============================================================
    # 新增：reset 增加课程学习支持
    # ============================================================
    def reset(self):
        """重置环境，支持课程学习动态调整任务难度"""
        self.target = [0, 0, 0]
        self.weapon = [0, 0, 0]
        self.raw_uavs = []
        self.episode_data = []
        self._episode_steps = 0
        self.right_vector = [[1, 0, 0] for i in range(self.n_total_uavs)]
        self.reward = None
        self.is_terminal = [False for _ in range(self.n_total_uavs)]

        # ---- 课程学习：调整任务成功率阈值 ----
        if self.mode == "train":
            self.current_stage_episodes += 1
            # 每20轮统计一次成功率，决定是否提升难度
            if self.current_stage_episodes >= 20:
                success_rate = self.current_stage_success_count / self.current_stage_episodes
                self.stage_success_history.append(success_rate)
                # 检查是否达到升级阈值
                if len(self.stage_success_history) >= 3:
                    avg_rate = np.mean(self.stage_success_history[-3:])
                    if avg_rate >= self.curriculum_thresholds[self.curriculum_stage]:
                        self.curriculum_stage = min(self.curriculum_stage + 1, len(self.curriculum_thresholds) - 1)
                        logger.info(f"课程学习升级到阶段 {self.curriculum_stage}，成功率 {avg_rate:.2f}")
                        self.stage_success_history = []
                self.current_stage_success_count = 0
                self.current_stage_episodes = 0

        # ---- 重置目标位置（课程学习影响武器距离） ----
        # 阶段0：武器较远（容易）
        # 阶段1：武器中等（适中）
        # 阶段2：武器较近（困难）
        stage_scale = 1.0 - 0.2 * self.curriculum_stage
        effective_weapon_dist = self.dis_target_weapon * stage_scale
        effective_weapon_dist = max(effective_weapon_dist, 100.0)  # 不低于100m

        self.target[0] = float(random.uniform(self.map.map_min_x + effective_weapon_dist,
                                              self.map.map_max_x - effective_weapon_dist))
        self.target[1] = float(random.uniform(self.map.map_min_y + effective_weapon_dist,
                                              self.map.map_max_y - effective_weapon_dist))
        self.target[2] = float(self.map.search_nh(self.target[0], self.target[1]) + self.coll_safe_dis)

        theta_w = 2 * math.pi * random.uniform(0, 1)
        self.weapon[0] = float(self.target[0] + effective_weapon_dist * math.cos(theta_w))
        self.weapon[1] = float(self.target[1] + effective_weapon_dist * math.sin(theta_w))
        self.weapon[2] = float(self.map.search_nh(self.weapon[0], self.weapon[1]) + self.coll_safe_dis)

        # ---- 重置无人机 ----
        for uav in range(self.n_total_uavs):
            uav_x, uav_y, uav_z = self._init_xyz()
            init_vel = self._init_toward_velocity([uav_x, uav_y, uav_z], self.target)
            temp_uav = TrainUAV(uav_x, uav_y, uav_z, *init_vel, AttackState.SAFE, UAVState.ALIVE)
            self.raw_uavs.append(temp_uav)

        if self.is_use_weapon:
            EnvironmentInterface.reset(self.n_total_uavs, self.weapon,
                                       [self.raw_uavs[i].velocity for i in range(self.n_total_uavs)],
                                       [self.raw_uavs[i].position for i in range(self.n_total_uavs)], self.map)

        which_idx = self._get_game_target_idx()
        data_save = {"uva_state": [x.to_dict() for x in self.raw_uavs],
                     "uva_actions": [-1 for _ in range(self.n_total_uavs)],
                     "_episode_steps": self._episode_steps, "reward": self.reward,
                     "right_vec": self.right_vector,
                     "c_target_id": id(self.raw_uavs[which_idx]) if which_idx is not None else "None",
                     "curriculum_stage": self.curriculum_stage}
        self.episode_data.append(data_save)

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

    def judge_random_position_available(self, uav_x, uav_y, uav_z):
        if uav_x < self.map.map_min_x or uav_x > self.map.map_max_x:
            return False
        if uav_y < self.map.map_min_y or uav_y > self.map.map_max_y:
            return False
        available = True
        if len(self.raw_uavs) > 0:
            for i in range(len(self.raw_uavs)):
                dis = compute_distance([uav_x, uav_y, uav_z], self.raw_uavs[i].position)
                if dis < self.uav_length + self.coll_safe_dis:
                    available = False
        return available

    # ============================================================
    # 重写：get_state_of_all_uav 入口保持不变
    # ============================================================
    def get_state_of_all_uav(self):
        return [self.get_observation_of_a_uav(i) for i in range(self.n_total_uavs)]

    # ============================================================
    # 核心改动：get_observation_of_a_uav（28维标准观测）
    # ============================================================
    def get_observation_of_a_uav(self, uav_id):
        """
        28维观测向量（单智能体）
        设计原则：不区分角色，通过 is_targeted 让网络自行区分

        维度索引：
        0:1   : 自身归一化坐标 (x, y)
        2:3   : 自身速度方向 (vx, vy) 归一化到 [-1,1]
        4:5   : 队友相对坐标 (dx, dy) 归一化
        6     : 与武器距离 / 3000.0
        7:8   : 武器炮口指向单位向量 (wx, wy)
        9     : 炮口与你方向的夹角余弦 (1=正面, -1=背面)
        10:11 : 最近威胁（子弹）相对坐标 (dx, dy)
        12    : 与最近威胁距离 / 500.0
        13    : 子弹预计到达时间 / 3.0
        14    : 是否被瞄准 (is_targeted)  0/1
        15    : 武器状态码 (0=NORMAL,1=TUNING,2=CAPTURE,3=FIRE) / 3.0
        16:27 : 预留（可放高度、地形等）
        """
        uav = self.raw_uavs[uav_id]
        pos = np.array(uav.position[:2])
        vel = np.array(uav.velocity[:2])

        # 速度归一化
        speed = np.linalg.norm(vel)
        if speed > 0:
            vel_norm = vel / speed
        else:
            vel_norm = np.array([0.0, 0.0])

        # 武器信息
        weapon_pos = np.array(self.weapon[:2])

        # ---- 临时：获取炮口朝向（真实接口替换这里） ----
        # TODO: 替换为 EnvironmentInterface.get_weapon_forward()
        weapon_forward = np.array([1.0, 0.0])

        # 计算关键信号
        rel_weapon = weapon_pos - pos
        dist_to_weapon = np.linalg.norm(rel_weapon)

        # 炮口与你的夹角
        if dist_to_weapon > 0:
            to_weapon = rel_weapon / dist_to_weapon
            angle_cos = np.dot(weapon_forward, to_weapon)
        else:
            angle_cos = 0.0

        # 是否被瞄准
        target_idx = self._get_game_target_idx()
        is_targeted = 1.0 if target_idx == uav_id else 0.0

        # 武器状态码（需要映射为 0~3）
        try:
            raw_state = EnvironmentInterface.get_weapon_state()
            state_map = {0: 0, 1: 1, 2: 2, 3: 3}  # NORMAL=0, TUNING=1, CAPTURE=2, FIRE=3
            weapon_state = state_map.get(raw_state, 0)
        except:
            weapon_state = 0

        # 队友相对坐标
        mate_pos = np.array([0.0, 0.0])
        for i in range(self.n_total_uavs):
            if i != uav_id and self.raw_uavs[i].status == UAVState.ALIVE:
                mate_pos = np.array(self.raw_uavs[i].position[:2]) - pos
                break

        # 子弹威胁（占位，后续接真实子弹列表）
        bullet_rel = np.array([0.0, 0.0])
        bullet_dist = 0.0
        bullet_timer = 0.0

        # 组装观测向量（28维）
        obs = []
        obs.extend(pos / self.map.map_max_x)           # 0:1
        obs.extend(vel_norm)                           # 2:3
        obs.extend(mate_pos / self.map.map_max_x)      # 4:5
        obs.append(dist_to_weapon / 3000.0)            # 6
        obs.extend(weapon_forward)                     # 7:8
        obs.append(angle_cos)                          # 9
        obs.extend(bullet_rel / self.map.map_max_x)    # 10:11
        obs.append(bullet_dist / 500.0)                # 12
        obs.append(bullet_timer / 3.0)                 # 13
        obs.append(is_targeted)                        # 14（灵魂信号）
        obs.append(weapon_state / 3.0)                 # 15
        obs.extend([0.0] * 12)                         # 16:27 预留

        return np.array(obs, dtype=np.float32)

    def _get_game_target_idx(self):
        c_target = EnvironmentInterface.try_get_current_target()
        if c_target is None:
            return None
        for i in range(self.n_total_uavs):
            if c_target is EnvironmentInterface.get_uav_list()[i]:
                return i
        return None

    # ============================================================
    # step 保持原有逻辑，只调整返回格式
    # ============================================================
    def step(self, action):
        actions = []
        for idx in range(self.n_total_uavs):
            action_index = int(np.where(action[idx] == 1)[0])
            actions.append([action_index])

        last_state = copy.deepcopy(self.raw_uavs)
        last_target = self._get_game_target_idx()
        lat_stat = EnvironmentInterface.get_weapon_state() if self.is_use_weapon else 0

        for idx in range(self.n_total_uavs):
            if self.raw_uavs[idx].status != UAVState.ALIVE:
                continue
            actual_action = actions[idx][0]
            radian_action = self.ACTION_SET[actual_action]

            temp = self.raw_uavs[idx]
            velocity = temp.velocity
            position = temp.position
            right = self.right_vector[idx]
            next_position, after_rotate_velocity_direction_in_parent, after_rotate_horizontal_right_vector_in_parent \
                = fly_from_9_selections(*radian_action, velocity, right, position)
            actual_velocity = (np.array(after_rotate_velocity_direction_in_parent) * self.uav_velocity_value).tolist()
            self.right_vector[idx] = after_rotate_horizontal_right_vector_in_parent
            self.raw_uavs[idx].set_position(*next_position)
            self.raw_uavs[idx].set_velocity(*actual_velocity)

        self._episode_steps += 1

        if self.is_use_weapon:
            position = [self.raw_uavs[u].position for u in range(self.n_total_uavs)]
            velocity = [self.raw_uavs[u].velocity for u in range(self.n_total_uavs)]
            game_uav_list = EnvironmentInterface.step(position, velocity)
            for u in range(self.n_total_uavs):
                if game_uav_list[u] is None:
                    self.raw_uavs[u].status = UAVState.DESTROYED

        self.set_reward(last_state, actions, last_target, lat_stat)

        ret_reward = [[x] for x in self.reward]
        return self.get_state_of_all_uav(), ret_reward, self.is_terminal, {i: {'individual_reward': self.reward[i]} for
                                                                           i in range(self.n_total_uavs)}

    def append_data(self, action):
        which_idx = self._get_game_target_idx()
        data_save = {"uva_state": [x.to_dict() for x in self.raw_uavs],
                     "uva_actions": action,
                     "right_vec": self.right_vector,
                     "_episode_steps": self._episode_steps,
                     "reward": self.reward,
                     "c_target_id": id(self.raw_uavs[which_idx]) if which_idx is not None else "None",
                     'r_msg': self.r_msg,
                     'degree': self.degree,
                     "curriculum_stage": self.curriculum_stage}
        self.episode_data.append(data_save)

    # ============================================================
    # 核心改动：set_reward（分离+共享奖励，含摸鱼/自杀惩罚）
    # ============================================================
    def set_reward(self, last_p, action, last_target, lat_stat):
        current_p = self.raw_uavs
        rewards = [0.0 for _ in range(self.n_total_uavs)]
        self.r_msg = ['' for _ in range(self.n_total_uavs)]
        self.degree = [None for _ in range(self.n_total_uavs)]

        # ===== 1. 全局共享奖励（循环外计算） =====
        r_shared_formation = 0.0
        if self.n_total_uavs >= 2:
            alive_positions = []
            for idx in range(self.n_total_uavs):
                if current_p[idx].status == UAVState.ALIVE:
                    alive_positions.append(np.array(current_p[idx].position[:2]))
            if len(alive_positions) >= 2:
                w_pos = np.array(self.weapon[:2])
                v0 = (alive_positions[0] - w_pos) / (np.linalg.norm(alive_positions[0] - w_pos) + 1e-8)
                v1 = (alive_positions[1] - w_pos) / (np.linalg.norm(alive_positions[1] - w_pos) + 1e-8)
                cos_angle = np.dot(v0, v1)
                # 夹角越接近180°（cos=-1），奖励越高
                r_shared_formation = 0.02 * (1 - cos_angle) / 2

        # 课程学习阶段奖励加成
        r_curriculum_bonus = self.curriculum_stage_rewards[self.curriculum_stage]

        r_task_success = 0.0
        for idx in range(self.n_total_uavs):
            if current_p[idx].status == UAVState.ALIVE:
                dist_to_target = compute_distance(current_p[idx].position, self.target)
                if dist_to_target <= self.task_success_radius:
                    r_task_success = 100.0 + r_curriculum_bonus
                    self.is_terminal = [True for _ in range(self.n_total_uavs)]
                    self.current_stage_success_count += 1
                    self.dump("任务完成！")
                    break

        # ===== 2. 个体分离奖励（循环内计算） =====
        for idx in range(self.n_total_uavs):
            uav = current_p[idx]
            if uav.status != UAVState.ALIVE:
                rewards[idx] = -10.0
                self.is_terminal[idx] = True
                self.r_msg[idx] = f'{idx}被摧毁/撞毁，'
                continue

            # ---- 基础生存惩罚 ----
            r_step = -0.01

            # ---- 核心角色信号 ----
            target_idx = self._get_game_target_idx()
            is_targeted = (target_idx == idx)
            dist_to_weapon = compute_distance(self.weapon, uav.position)

            # ---- 诱饵/刺客分支奖励 ----
            r_bait = 0.0
            if is_targeted:
                # 诱饵：保持在安全边界（1600~1900m）
                if 1600 < dist_to_weapon < 1900:
                    r_bait = 0.02
                else:
                    r_bait = -0.02 * (abs(dist_to_weapon - 1750) / 200)

                # 站桩惩罚：切向速度太小
                vel = np.array(uav.velocity[:2])
                if np.linalg.norm(vel) > 0:
                    to_weapon = (np.array(self.weapon[:2]) - np.array(uav.position[:2]))
                    to_weapon_norm = np.linalg.norm(to_weapon)
                    if to_weapon_norm > 0:
                        to_weapon = to_weapon / to_weapon_norm
                        lateral_speed = np.linalg.norm(vel - np.dot(vel, to_weapon) * to_weapon)
                        if lateral_speed < 0.2 * self.uav_velocity_value:
                            r_bait -= 0.02

                self.r_msg[idx] += f'诱饵(dist={dist_to_weapon:.0f}, lat={lateral_speed:.1f}), '
            else:
                # 刺客：靠近武器有奖
                last_dist = compute_distance(self.weapon, last_p[idx].position) if last_p else dist_to_weapon
                r_bait = 0.02 * (last_dist - dist_to_weapon) / self.uav_velocity_value
                r_bait = np.clip(r_bait, -0.05, 0.05)

                # 摸鱼惩罚：队友被锁，你却在远处看戏
                if target_idx is not None and target_idx != idx and dist_to_weapon > 1800:
                    r_bait -= 0.1
                    self.r_msg[idx] += '摸鱼惩罚, '

                self.r_msg[idx] += f'刺客(dist={dist_to_weapon:.0f}), '

            # ---- 自杀惩罚：冲进1500m死亡线 ----
            r_suicide = 0.0
            if dist_to_weapon < 1500:
                r_suicide = -0.5
                self.r_msg[idx] += '自杀警告, '

            # ---- 汇总个体奖励 ----
            rewards[idx] = r_step + r_bait + r_suicide

            # ---- 加上共享奖励 ----
            rewards[idx] += r_shared_formation + r_task_success

            # ---- 截断 ----
            rewards[idx] = np.clip(rewards[idx], -2.0, 2.0)

            # 日志记录
            self.r_msg[idx] += f'reward={rewards[idx]:.2f}'

        self.reward = rewards

        # ===== 3. 终局判定 =====
        if r_task_success > 0:
            self.append_data(action)
            msg = "任务完成！"
            self.dump(msg)
            green_str = _green_log_str(msg)
            logger.info(f"PID-{os.getpid()}, mode-{self.mode}, episode-{self.n_episode} {green_str}")
            self.n_episode += 1
            return

        if all(self.is_terminal):
            self.append_data(action)
            msg = "都被击毁！"
            self.dump(msg)
            logger.info(f"PID-{os.getpid()}, mode-{self.mode}, episode-{self.n_episode} {msg}")
            self.n_episode += 1
            return

        if self._episode_steps >= self.max_episode_steps:
            for i in range(self.n_total_uavs):
                self.r_msg[i] += '步子到了。'
            self.n_episode += 1
            self.is_terminal = [True for _ in range(self.n_total_uavs)]
            logger.info(f"PID-{os.getpid()}, mode-{self.mode}, episode-{self.n_episode} [terminated]：超出最大限制")
            self.append_data(action)
            self.dump("超出最大限制")
            return

        self.append_data(action)

    def compute_init_velocity(self):
        speed = self.uav_velocity_value
        yaw = random.uniform(0, 2 * math.pi)
        pitch = random.uniform(-math.pi / 2, math.pi / 2)
        dx = math.cos(pitch) * math.cos(yaw)
        dy = math.cos(pitch) * math.sin(yaw)
        dz = math.sin(pitch)
        init_vel = [dx * speed, dy * speed, dz * speed]
        return init_vel

    def seed(self, seed_num=None):
        pass

    def get_observation_size_of_a_uav(self):
        # 28维 + 地形预留
        if self.is_use_weapon:
            return 28
        else:
            return 3 + 3 + 3 + 3 + 3

    def close(self):
        pass