"""
测试无人机横向机动躲避子弹的能力
无人机在 1500m 边界做横向正弦摆动，观察能否躲避武器开火
"""

import numpy as np
import time
import configparser
import math
from pathlib import Path

from trainer.drone.mul_uav_env import MultiUavEnv
from trainer.utils.util import compute_distance
from trainer.drone.weapons.interfaces.environment_interface import EnvironmentInterface
from trainer.drone.weapons.entries.uav.uav_enum import UAVState, AttackState


class DodgeTester:
    def __init__(self, speed=150.0, amplitude=30.0, frequency=0.8, step_limit=300):
        """
        :param speed: 无人机速度 (m/s)
        :param amplitude: 横向摆动幅度 (m) 即偏离 1500m 的距离
        :param frequency: 摆动频率 (Hz)
        :param step_limit: 最大步数
        """
        self.speed = speed
        self.amplitude = amplitude
        self.frequency = frequency
        self.step_limit = step_limit
        self.env = None
        self.weapon_state_names = ['NORMAL', 'TUNING', 'CAPTURE', 'FIRE', 'RELOAD']

    def _get_weapon_state_name(self, state_code):
        if 0 <= state_code < len(self.weapon_state_names):
            return self.weapon_state_names[state_code]
        return 'UNKNOWN'

    def setup(self):
        config_path = '../drone/config/th_demo.ini'   # 根据实际路径调整
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
        print("🎯 横向机动闪避测试")
        print("=" * 70)
        print(f"📌 目标位置: {self.target}")
        print(f"🔫 武器位置: {self.weapon_pos}")
        print(f"⚡ 无人机速度: {self.speed} m/s")
        print(f"📐 摆动幅度: {self.amplitude} m")
        print(f"🔄 摆动频率: {self.frequency} Hz")
        print("=" * 70)
        return obs

    def compute_dodge_action(self, uav_pos, step):
        """
        计算横向摆动动作：
        保持与武器距离在 1500m 附近，同时沿切向做正弦摆动。
        返回三维动作 [vx, vy, vz]，vz = 0（水平面飞行）
        """
        rel = uav_pos[:2] - self.weapon_pos[:2]
        dist = np.linalg.norm(rel)
        if dist < 1:
            angle = 0.0
        else:
            angle = math.atan2(rel[1], rel[0])

        # 横向位移：在切向方向上的偏移量
        t = step * 0.1  # DT = 0.1
        lateral_offset = self.amplitude * math.sin(2 * math.pi * self.frequency * t)

        # 目标距离保持 1500m（武器开火线附近）
        target_dist = 1500.0

        # 目标角度 = 当前角度 + 横向位移 / 半径（小角度近似）
        target_angle = angle + lateral_offset / target_dist

        # 目标位置
        target_pos = self.weapon_pos[:2] + target_dist * np.array([math.cos(target_angle), math.sin(target_angle)])

        # 速度方向指向目标位置
        delta = target_pos - uav_pos[:2]
        norm = np.linalg.norm(delta)
        if norm < 1:
            return np.array([0.0, 0.0, 0.0])
        direction = delta / norm

        # 速度指令（限制最大速度）
        max_speed = self.env.uav_velocity_value
        action_xy = direction * (self.speed / max_speed)
        action_xy = np.clip(action_xy, -1.0, 1.0)
        # 返回三维向量，z 分量为 0
        return np.array([action_xy[0], action_xy[1], 0.0])

    def run(self):
        self.setup()

        # 只控制第一架无人机，第二架移到战场外并悬停
        uav = self.env.raw_uavs[0]
        if self.num_agents > 1:
            self.env.raw_uavs[1].position = [self.env.map.map_max_x - 100,
                                             self.env.map.map_max_y - 100,
                                             self.env.raw_uavs[1].position[2]]
            self.env.raw_uavs[1].velocity = [0, 0, 0]

        step = 0
        hit = False
        state_history = []

        print("\n🚀 开始闪避测试...\n")

        while step < self.step_limit:
            # 计算横向摆动动作（三维）
            action_vec = self.compute_dodge_action(uav.position, step)
            action_list = [action_vec.tolist()]
            if self.num_agents > 1:
                action_list.append([0.0, 0.0, 0.0])

            obs, rewards, done, info = self.env.step(action_list)

            # 检查无人机状态
            uav = self.env.raw_uavs[0]
            if uav.status != UAVState.ALIVE:
                print(f"\n💥 无人机在第 {step} 步被击毁！")
                hit = True
                break

            weapon_state = self.get_weapon_state()
            state_name = self._get_weapon_state_name(weapon_state)
            dist_to_weapon = compute_distance(self.weapon_pos, uav.position)
            bullet_count = self.get_bullet_count()

            if step % 10 == 0:
                print(f"[{step:3d}] 武器状态: {state_name}, 距离: {dist_to_weapon:.1f}m, 子弹数: {bullet_count}")

            state_history.append(weapon_state)
            step += 1

        if not hit:
            print(f"\n✅ 无人机存活了 {step} 步（达到最大步数）")

        self.print_summary(state_history, hit, step)

        print(f"\n📁 完整数据已保存到: {self.env.episode_data_file}")
        print("💡 用可视化 HTML 加载此 JSON 文件查看轨迹")

        self.env.close()
        return self.env.episode_data_file

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

    def print_summary(self, state_history, hit, steps):
        print("\n" + "=" * 70)
        print("📊 测试总结")
        print("=" * 70)
        print(f"总步数: {steps}")
        print(f"被击毁: {'是' if hit else '否'}")
        print(f"武器状态分布:")
        states = state_history
        state_counts = {s: states.count(s) for s in set(states)}
        for s, count in sorted(state_counts.items()):
            name = self._get_weapon_state_name(s)
            print(f"  {name}: {count} 步")
        print("=" * 70)


def main():
    # 您可以调整这些参数来测试不同速度、幅度和频率下的躲避效果
    tester = DodgeTester(
        speed=150.0,        # 速度 150 m/s
        amplitude=30.0,     # 摆动幅度 ±30m（相对于1500m线）
        frequency=0.8,      # 0.8 Hz 摆动频率
        step_limit=300
    )
    tester.run()


if __name__ == "__main__":
    main()