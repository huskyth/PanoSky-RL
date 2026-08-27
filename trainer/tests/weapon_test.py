"""
武器系统测试脚本 - 两架无人机同时飞向目标
验证多目标跟踪、目标切换、调弦/捕获/开火等
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
    def __init__(self, speed1=80.0, speed2=60.0, step_limit=300, log_interval=10):
        self.speed1 = speed1          # 第一架无人机速度 (m/s)
        self.speed2 = speed2          # 第二架无人机速度 (m/s)
        self.step_limit = step_limit
        self.log_interval = log_interval
        self.env = None
        self.weapon_state_names = ['NORMAL', 'TUNING', 'CAPTURE', 'FIRE', 'RELOAD']
        self.total_steps = 0

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

        self.env = MultiUavEnv(
            rank=0,
            mode="test",
            cf=cf,
            episode_limit=self.step_limit,
            is_debug=True,
            is_share=True,
            is_use_weapon=True
        )

        obs = self.env.reset()
        self.target = np.array(self.env.target)
        self.weapon_pos = np.array(self.env.weapon)
        self.num_agents = self.env.n_total_uavs

        Path("./visualization_data").mkdir(exist_ok=True)

        print("=" * 70)
        print("🎯 武器系统测试（多目标）开始")
        print("=" * 70)
        print(f"📌 目标位置: {self.target}")
        print(f"🔫 武器位置: {self.weapon_pos}")
        print(f"📁 数据文件: {self.env.episode_data_file}")
        print(f"⚡ UAV-0 速度: {self.speed1} m/s")
        print(f"⚡ UAV-1 速度: {self.speed2} m/s")
        print("=" * 70)
        return obs

    def compute_action(self, uav_pos, speed):
        """计算指向目标的动作向量"""
        direction = self.target - np.array(uav_pos)
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0, 0.0])
        max_speed = self.env.uav_velocity_value
        action = direction * (speed / max_speed)
        return np.clip(action, -1.0, 1.0)

    def get_weapon_state(self):
        try:
            return EnvironmentInterface.get_weapon_state()
        except:
            return -1

    def get_bullet_count(self):
        try:
            bullets = EnvironmentInterface.get_bullets()
            return len(bullets) if bullets else 0
        except:
            return 0

    def run(self):
        self.setup()

        # 初始化动作
        actions = []
        for i in range(self.num_agents):
            if i == 0:
                speed = self.speed1
            else:
                speed = self.speed2
            action_vec = self.compute_action(self.env.raw_uavs[i].position, speed)
            actions.append(action_vec.tolist())

        step = 0
        last_state = -1
        state_history = []

        print("\n🚀 两架无人机开始飞行...\n")

        while step < self.step_limit:
            obs, rewards, done, info = self.env.step(actions)

            # 获取两架无人机状态
            uav0 = self.env.raw_uavs[0]
            uav1 = self.env.raw_uavs[1]
            pos0 = np.array(uav0.position)
            pos1 = np.array(uav1.position)

            dist0_target = compute_distance(pos0, self.target)
            dist1_target = compute_distance(pos1, self.target)
            dist0_weapon = compute_distance(pos0, self.weapon_pos)
            dist1_weapon = compute_distance(pos1, self.weapon_pos)

            weapon_state = self.get_weapon_state()
            state_name = self._get_weapon_state_name(weapon_state)
            target_idx = self.env._get_game_target_idx()
            is_locked_0 = (target_idx == 0)
            is_locked_1 = (target_idx == 1)

            bullet_count = self.get_bullet_count()
            attack_state0 = getattr(uav0, 'attacked_state', AttackState.SAFE)
            attack_state1 = getattr(uav1, 'attacked_state', AttackState.SAFE)

            # 打印信息
            if step % self.log_interval == 0 or weapon_state != last_state:
                print(f"[{step:3d}] 状态: {state_name:>8} | "
                      f"锁定: {'UAV-0' if is_locked_0 else 'UAV-1' if is_locked_1 else '无'} | "
                      f"子弹: {bullet_count:2d}")
                print(f"        UAV-0: 目标距离 {dist0_target:6.1f}m, 武器距离 {dist0_weapon:6.1f}m, "
                      f"状态 {self._get_uav_status_name(uav0.status)}")
                print(f"        UAV-1: 目标距离 {dist1_target:6.1f}m, 武器距离 {dist1_weapon:6.1f}m, "
                      f"状态 {self._get_uav_status_name(uav1.status)}")
                last_state = weapon_state

            state_history.append(weapon_state)

            # 判断结束（任意一架被击毁或到达目标）
            if uav0.status == UAVState.DESTROYED or attack_state0 == AttackState.DESTROYED:
                print(f"\n💥 UAV-0 被击毁！")
                break
            if uav1.status == UAVState.DESTROYED or attack_state1 == AttackState.DESTROYED:
                print(f"\n💥 UAV-1 被击毁！")
                break

            if dist0_target <= self.env.task_success_radius or dist1_target <= self.env.task_success_radius:
                print(f"\n✅ 有无人机到达目标！")
                print(f"   UAV-0 距离: {dist0_target:.1f}m")
                print(f"   UAV-1 距离: {dist1_target:.1f}m")
                break

            # 更新动作（方向可能变化）
            for i in range(self.num_agents):
                if self.env.raw_uavs[i].status == UAVState.ALIVE:
                    speed = self.speed1 if i == 0 else self.speed2
                    action_vec = self.compute_action(self.env.raw_uavs[i].position, speed)
                    actions[i] = action_vec.tolist()

            step += 1

        self.total_steps = step + 1
        self.print_summary(state_history)

        print(f"\n📁 完整数据已保存到: {self.env.episode_data_file}")
        print("💡 用可视化 HTML 加载此 JSON 文件查看轨迹")

        self.env.close()
        return self.env.episode_data_file

    def print_summary(self, state_history):
        print("\n" + "=" * 70)
        print("📊 测试总结")
        print("=" * 70)

        uav0 = self.env.raw_uavs[0]
        uav1 = self.env.raw_uavs[1]
        pos0 = np.array(uav0.position)
        pos1 = np.array(uav1.position)
        dist0 = compute_distance(pos0, self.target)
        dist1 = compute_distance(pos1, self.target)

        print(f"总步数: {self.total_steps}")
        print(f"UAV-0 状态: {self._get_uav_status_name(uav0.status)}, 距目标: {dist0:.1f}m")
        print(f"UAV-1 状态: {self._get_uav_status_name(uav1.status)}, 距目标: {dist1:.1f}m")

        states = state_history
        state_counts = {s: states.count(s) for s in set(states)}
        print("\n武器状态分布:")
        for s, count in sorted(state_counts.items()):
            name = self._get_weapon_state_name(s)
            print(f"  {name}: {count} 步")

        if 0 in states and 1 in states and 2 in states:
            print("\n✅ 武器流程完整: NORMAL → TUNING → CAPTURE")
            if 3 in states:
                print("✅ 进入 FIRE 状态（开火）")
            else:
                print("⚠️ 未进入 FIRE 状态")
        else:
            print("\n⚠️ 武器流程不完整")

        print("=" * 70)


def main():
    tester = WeaponTester(
        speed1=100.0,   # 第一架无人机速度 100 m/s
        speed2=70.0,    # 第二架无人机速度 70 m/s (稍慢，形成错位)
        step_limit=500,
        log_interval=20
    )
    tester.run()


if __name__ == "__main__":
    main()