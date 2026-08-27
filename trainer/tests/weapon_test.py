"""
武器系统测试脚本 - 直接飞向目标
直接复用环境自带的 _record_step_data 记录数据
"""

import numpy as np
import time
import configparser
import json
from pathlib import Path
import subprocess
import sys

from trainer.drone.mul_uav_env import MultiUavEnv
from trainer.utils.util import compute_distance
from trainer.drone.weapons.interfaces.environment_interface import EnvironmentInterface
from trainer.drone.weapons.entries.uav.uav_enum import UAVState, AttackState


class WeaponTester:
    def __init__(self, speed=80.0, step_limit=300, use_uav_index=0, log_interval=10):
        self.speed = speed
        self.step_limit = step_limit
        self.uav_idx = use_uav_index
        self.log_interval = log_interval
        self.env = None
        self.weapon_state_names = ['NORMAL', 'TUNING', 'CAPTURE', 'FIRE', 'RELOAD']
        self.total_steps = 0  # 用于存储总步数

    def _get_weapon_state_name(self, state_code):
        if 0 <= state_code < len(self.weapon_state_names):
            return self.weapon_state_names[state_code]
        return 'UNKNOWN'

    def _get_uav_status_name(self, status):
        if status == UAVState.ALIVE:
            return 'ALIVE'
        elif status == UAVState.DESTROYED:
            return 'DESTROYED'
        elif status == UAVState.COLLISION:
            return 'COLLISION'
        return 'UNKNOWN'

    def setup(self):
        config_path = '../drone/config/th_demo.ini'
        cf = configparser.ConfigParser()
        cf.read(config_path, encoding='utf-8')

        # is_debug=True 会自动生成 JSON 数据文件
        self.env = MultiUavEnv(
            rank=0,
            mode="test",
            cf=cf,
            episode_limit=self.step_limit,
            is_debug=True,          # 开启数据记录（会生成 JSON）
            is_share=True,
            is_use_weapon=True
        )

        obs = self.env.reset()
        self.target = np.array(self.env.target)
        self.weapon_pos = np.array(self.env.weapon)
        self.num_agents = self.env.n_total_uavs

        # 确保目录存在
        Path("./visualization_data").mkdir(exist_ok=True)

        print("=" * 70)
        print("🎯 武器系统测试开始")
        print("=" * 70)
        print(f"📌 目标位置: {self.target}")
        print(f"🔫 武器位置: {self.weapon_pos}")
        print(f"📁 数据文件: {self.env.episode_data_file}")
        print(f"⚡ 飞行速度: {self.speed} m/s (最大: {self.env.uav_velocity_value} m/s)")
        print("=" * 70)
        return obs

    def compute_action(self, uav_pos):
        direction = self.target - np.array(uav_pos)
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0, 0.0])
        max_speed = self.env.uav_velocity_value
        action = direction * (self.speed / max_speed)
        return np.clip(action, -1.0, 1.0)

    def get_weapon_state(self):
        try:
            return EnvironmentInterface.get_weapon_state()
        except Exception as e:
            return -1

    def get_bullet_count(self):
        try:
            bullets = EnvironmentInterface.get_bullets()
            return len(bullets) if bullets else 0
        except:
            return 0

    def run(self):
        self.setup()

        uav = self.env.raw_uavs[self.uav_idx]

        # 另一架无人机悬停在地图角落
        if self.num_agents > 1:
            for i in range(self.num_agents):
                if i != self.uav_idx:
                    self.env.raw_uavs[i].position = [self.env.map.map_max_x - 100,
                                                     self.env.map.map_max_y - 100,
                                                     self.env.raw_uavs[i].position[2]]
                    self.env.raw_uavs[i].velocity = [0, 0, 0]

        action_vec = self.compute_action(uav.position)
        actions = []
        for i in range(self.num_agents):
            if i == self.uav_idx:
                actions.append(action_vec.tolist())
            else:
                actions.append([0.0, 0.0, 0.0])

        step = 0
        last_state = -1
        state_history = []

        print("\n🚀 开始飞行...\n")

        while step < self.step_limit:
            obs, rewards, done, info = self.env.step(actions)
            uav = self.env.raw_uavs[self.uav_idx]
            uav_pos = np.array(uav.position)

            dist_to_target = compute_distance(uav_pos, self.target)
            dist_to_weapon = compute_distance(uav_pos, self.weapon_pos)

            weapon_state = self.get_weapon_state()
            state_name = self._get_weapon_state_name(weapon_state)
            target_idx = self.env._get_game_target_idx()
            is_locked = (target_idx == self.uav_idx)

            bullet_count = self.get_bullet_count()
            attack_state = getattr(uav, 'attacked_state', AttackState.SAFE)

            # 仅在控制台打印（不重复存大量数据，环境已存 JSON）
            if step % self.log_interval == 0 or weapon_state != last_state:
                print(f"[{step:3d}] 目标距离: {dist_to_target:6.1f}m | "
                      f"状态: {state_name:>8} | "
                      f"锁定: {'✅' if is_locked else '❌'} | "
                      f"子弹: {bullet_count:2d} | "
                      f"UAV: {self._get_uav_status_name(uav.status)}")
                last_state = weapon_state

            state_history.append(weapon_state)

            # 判断结束
            if uav.status == UAVState.DESTROYED or attack_state == AttackState.DESTROYED:
                print(f"\n💥 无人机 {self.uav_idx} 被击毁！")
                break

            if dist_to_target <= self.env.task_success_radius:
                print(f"\n✅ 无人机到达目标！距离: {dist_to_target:.1f}m")
                break

            step += 1

        # 保存总步数（step 是最后一步的索引，总步数 = step + 1）
        self.total_steps = step + 1

        # 打印总结
        self.print_summary(state_history)

        # 提示数据文件位置
        print(f"\n📁 完整数据已保存到: {self.env.episode_data_file}")
        print("💡 用可视化 HTML 加载此 JSON 文件查看轨迹")

        self.env.close()
        return self.env.episode_data_file

    def print_summary(self, state_history):
        print("\n" + "=" * 70)
        print("📊 测试总结")
        print("=" * 70)

        uav = self.env.raw_uavs[self.uav_idx]
        uav_pos = np.array(uav.position)
        dist_to_target = compute_distance(uav_pos, self.target)

        print(f"总步数: {self.total_steps}")
        print(f"无人机状态: {self._get_uav_status_name(uav.status)}")
        print(f"最终距目标: {dist_to_target:.1f}m")

        # 武器状态分布
        states = state_history
        state_counts = {s: states.count(s) for s in set(states)}
        print("\n武器状态分布:")
        for s, count in sorted(state_counts.items()):
            name = self._get_weapon_state_name(s)
            print(f"  {name}: {count} 步")

        # 流程检查
        if 0 in states and 1 in states and 2 in states:
            print("\n✅ 武器流程完整: NORMAL → TUNING → CAPTURE")
            if 3 in states:
                print("✅ 进入 FIRE 状态（开火）")
            else:
                print("⚠️ 未进入 FIRE 状态（可能需要更靠近或更长时间）")
        else:
            print("\n⚠️ 武器流程不完整，检查无人机是否进入射程")

        print("=" * 70)


def main():
    tester = WeaponTester(
        speed=100.0,
        step_limit=500,
        use_uav_index=0,
        log_interval=10
    )
    tester.run()


if __name__ == "__main__":
    main()