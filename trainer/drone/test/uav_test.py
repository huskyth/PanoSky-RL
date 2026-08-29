from pathlib import Path
import sys
import numpy as np
from trainer.drone.mul_uav_env import MultiUavEnv
import configparser

# 加载配置文件
from trainer.drone.weapons.entries.uav.uav_enum import UAVState

config_path = Path(__file__).parent.parent / 'config/th_demo.ini'
cf = configparser.ConfigParser()
cf.read(str(config_path), encoding='utf-8')

# 创建环境（这里可以开关武器，is_use_weapon=True 武器会攻击，False则纯飞行）
env = MultiUavEnv(
    cf=cf,
    is_debug=True,
    episode_limit=500,
    rank=0,
    is_use_weapon=True,  # 设为 False 可避免被击毁，便于观察飞行
    is_share=True
)

obs = env.reset()
target = np.array(env.target)
num_agents = env.n_total_uavs
max_speed = env.uav_velocity_value

print("=" * 60)
print(f"目标位置: {target}")
print(f"无人机数量: {num_agents}")
print(f"最大速度: {max_speed} m/s")
print("开始贪心飞行（连续动作）...")
print("=" * 60)

step = 0
while True:
    actions = []
    for i in range(num_agents):
        # 如果无人机已死亡，动作置零
        if env.raw_uavs[i].status != UAVState.ALIVE:  # 0 代表 ALIVE
            actions.append([0.0, 0.0, 0.0])
            continue

        # 当前位置
        pos = np.array(env.raw_uavs[i].position)
        # 指向目标的单位向量
        dir_vec = target - pos
        norm = np.linalg.norm(dir_vec)
        if norm > 1e-6:
            dir_vec = dir_vec / norm
        else:
            dir_vec = np.array([0.0, 0.0, 0.0])
        # 动作 = 方向向量（范围 [-1,1]），乘以 1.0 表示全速
        actions.append(dir_vec.tolist())

    # 执行一步
    obs, rewards, done, info = env.step(np.array(actions))
    step += 1

    # 每 10 步打印距离
    if step % 10 == 0:
        for i in range(num_agents):
            if env.raw_uavs[i].status == 0:
                dist = np.linalg.norm(np.array(env.raw_uavs[i].position) - target)
                print(f"Step {step:4d} | UAV-{i} 距目标: {dist:7.2f} m")

    # 结束条件
    if all(env.is_terminal):
        print("\nEpisode 终止（任务完成或全部阵亡）。")
        break
    if step >= 500:
        print("达到最大步数，强制结束。")
        break

print("\n" + "=" * 60)
print("最终状态:")
for i in range(num_agents):
    if env.raw_uavs[i].status == 0:
        pos = np.array(env.raw_uavs[i].position)
        dist = np.linalg.norm(pos - target)
        print(f"UAV-{i}: 位置 {pos}, 距目标 {dist:.2f} m, 存活")
    else:
        print(f"UAV-{i}: 已阵亡")
print("=" * 60)

env.close()