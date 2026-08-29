from pathlib import Path
import sys
import numpy as np
import math
import time
from trainer.drone.mul_uav_env import MultiUavEnv
import configparser

# 加载配置
from trainer.drone.weapons.entries.uav.uav_enum import UAVState

config_path = Path(__file__).parent.parent / 'drone/config/th_demo.ini'
cf = configparser.ConfigParser()
cf.read(str(config_path), encoding='utf-8')

# 创建环境（开启武器）
env = MultiUavEnv(
    cf=cf,
    is_debug=True,
    episode_limit=500,
    rank=0,
    is_use_weapon=True,
    is_share=True
)

obs = env.reset()
target = np.array(env.target)
weapon_pos = np.array(env.weapon)
num_agents = env.n_total_uavs

print(f"目标位置: {target}")
print(f"武器位置: {weapon_pos}")

# 只控制第一架无人机，第二架移到角落
if num_agents > 1:
    env.raw_uavs[1].position = [env.map.map_max_x - 100,
                                env.map.map_max_y - 100,
                                env.raw_uavs[1].position[2]]
    env.raw_uavs[1].velocity = [0, 0, 0]

uav = env.raw_uavs[0]
# 初始位置：武器前方 2000m 处
start_pos = weapon_pos[:2] + 2000 * np.array([1.0, 0.0])
uav.position = [start_pos[0], start_pos[1], 0.0]
uav.velocity = [0, 0, 0]

step = 0
phase = "APPROACH"  # APPROACH -> CIRCLE -> DIVE
radius = 1500.0
angle = 0.0
fire_detected = False
reload_detected = False
max_steps = 500

print("=" * 70)
print("战术路径：接近 → 被锁定时圆周机动 → 装弹后冲刺")
print(f"目标位置: {target}")
print(f"武器位置: {weapon_pos}")
print("=" * 70)

while step < max_steps:
    # ---- 获取当前状态 ----
    uav_pos = np.array(uav.position)
    dist_to_weapon = np.linalg.norm(uav_pos[:2] - weapon_pos[:2])
    dist_to_target = np.linalg.norm(uav_pos - target)

    # 获取武器状态（环境内部方法）
    try:
        from trainer.drone.weapons.interfaces.environment_interface import EnvironmentInterface
        weapon_state = EnvironmentInterface.get_weapon_state()
    except:
        weapon_state = 0

    # ---- 检测开火和装弹 ----
    if weapon_state == 3 and not fire_detected:
        fire_detected = True
        print(f"[{step}] 🔥 武器开火！进入圆周机动 (距离武器: {dist_to_weapon:.1f}m)")

    if fire_detected and weapon_state != 3:
        reload_detected = True
        print(f"[{step}] 🚀 武器装弹/冷却！开始冲刺 (距离武器: {dist_to_weapon:.1f}m)")

    # ---- 阶段决策（确保每步都有动作） ----
    if not fire_detected:
        # 阶段1：直线接近武器
        dir_vec = weapon_pos[:2] - uav_pos[:2]
        norm = np.linalg.norm(dir_vec)
        if norm > 0:
            dir_vec = dir_vec / norm
        action = np.array([dir_vec[0], dir_vec[1], 0.0])
        phase = "APPROACH"

    elif fire_detected and not reload_detected:
        # 阶段2：圆周机动 —— 动作方向与无人机–武器连线垂直（纯切向）
        rel = uav_pos[:2] - weapon_pos[:2]
        dist = np.linalg.norm(rel)
        if dist < 1:
            rel = np.array([1.0, 0.0])
            dist = 1.0

        # 径向单位向量（从武器指向无人机）
        rad = rel / dist
        # 切向单位向量（逆时针垂直），即与连线垂直
        tan = np.array([-rad[1], rad[0]])

        # 纯切向动作，不添加径向修正，速度方向始终垂直
        action_dir = tan
        norm = np.linalg.norm(action_dir)
        if norm > 0:
            action_dir = action_dir / norm
        action = np.array([action_dir[0], action_dir[1], 0.0])
        phase = "CIRCLE"

        # 打印每10步的状态
        if step % 10 == 0:
            print(f"[{step}] 🔄 圆周机动 | 半径: {dist:.1f}m | 武器状态: {weapon_state}")

    else:
        # 阶段3：冲刺目标
        dir_vec = target[:2] - uav_pos[:2]
        norm = np.linalg.norm(dir_vec)
        if norm > 0:
            dir_vec = dir_vec / norm
        action = np.array([dir_vec[0], dir_vec[1], 0.0])
        phase = "DIVE"
        if step % 5 == 0:
            print(f"[{step}] 💨 冲刺 | 距目标: {dist_to_target:.1f}m")

    # ---- 执行一步 ----
    actions = [action.tolist()]
    if num_agents > 1:
        actions.append([0.0, 0.0, 0.0])

    obs, rewards, done, info = env.step(np.array(actions))
    step += 1

    # ---- 检查结束条件 ----
    if uav.status != UAVState.ALIVE:
        print(f"[{step}] 💥 无人机被击毁！")
        break

    if dist_to_target <= env.task_success_radius:
        print(f"[{step}] ✅ 到达目标！距离: {dist_to_target:.1f}m")
        break

    if step >= max_steps:
        print(f"[{step}] ⏰ 达到最大步数")
        break

# ---- 最终总结 ----
print("\n" + "=" * 70)
print("📊 最终状态:")
print(f"  无人机位置: {uav.position}")
print(f"  距离武器: {np.linalg.norm(np.array(uav.position[:2]) - weapon_pos[:2]):.1f}m")
print(f"  距离目标: {np.linalg.norm(np.array(uav.position) - target):.1f}m")
print(f"  总步数: {step}")
print("=" * 70)

env.close()