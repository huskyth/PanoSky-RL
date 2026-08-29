"""
测试无人机在被武器锁定并开火后，通过圆周运动能否逃脱
先悬停等待开火，然后切换为圆周运动
"""

import numpy as np
import configparser
import math
import time
from pathlib import Path

from trainer.drone.mul_uav_env import MultiUavEnv
from trainer.utils.util import compute_distance
from trainer.drone.weapons.interfaces.environment_interface import EnvironmentInterface
from trainer.drone.weapons.entries.uav.uav_enum import UAVState, AttackState


class EscapeAfterFireTester:
    def __init__(self, speed=250.0, radius=1500.0, step_limit=300,
                 wait_steps=50, start_dist=1450):
        """
        :param speed: 圆周切向速度 (m/s)
        :param radius: 圆周半径 (m)
        :param step_limit: 最大步数
        :param wait_steps: 悬停等待开火的最大步数
        :param start_dist: 初始距离 (m)
        """
        self.speed = speed
        self.radius = radius
        self.step_limit = step_limit
        self.wait_steps = wait_steps
        self.start_dist = start_dist
        self.env = None
        self.weapon_state_names = ['NORMAL', 'TUNING', 'CAPTURE', 'FIRE', 'RELOAD']

    def _get_weapon_state_name(self, state_code):
        if 0 <= state_code < len(self.weapon_state_names):
            return self.weapon_state_names[state_code]
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
        print("🎯 开火后圆周逃逸测试")
        print("=" * 70)
        print(f"🔫 武器位置: {self.weapon_pos}")
        print(f"⚡ 切向速度: {self.speed} m/s")
        print(f"⭕ 圆周半径: {self.radius} m")
        print(f"📍 初始距离: {self.start_dist} m")
        print(f"⏳ 等待开火时间: {self.wait_steps * 0.1}s")
        print("=" * 70)
        return obs

    def compute_circular_action(self, uav_pos, step):
        """计算匀速圆周运动的速度指令"""
        rel = uav_pos[:2] - self.weapon_pos[:2]
        dist = np.linalg.norm(rel)
        if dist < 1:
            angle = 0.0
        else:
            angle = math.atan2(rel[1], rel[0])

        # 保持半径恒定，沿切向（逆时针）
        rad_vec = rel / dist
        tan_vec = np.array([-rad_vec[1], rad_vec[0]])
        direction = tan_vec

        max_speed = self.env.uav_velocity_value
        action_xy = direction * (self.speed / max_speed)
        action_xy = np.clip(action_xy, -1.0, 1.0)

        return np.array([action_xy[0], action_xy[1], 0.0])

    def compute_hold_action(self, uav_pos):
        """悬停（速度为零）"""
        return np.array([0.0, 0.0, 0.0])

    def run(self):
        self.setup()

        uav = self.env.raw_uavs[0]
        # 把无人机放到武器前方 start_dist 处
        start_angle = 0.0
        start_pos = self.weapon_pos[:2] + self.start_dist * np.array([math.cos(start_angle), math.sin(start_angle)])
        uav.position = [start_pos[0], start_pos[1], 0.0]
        uav.velocity = [0, 0, 0]

        if self.num_agents > 1:
            self.env.raw_uavs[1].position = [self.env.map.map_max_x - 100,
                                             self.env.map.map_max_y - 100,
                                             self.env.raw_uavs[1].position[2]]
            self.env.raw_uavs[1].velocity = [0, 0, 0]

        step = 0
        hit = False
        state_history = []
        dist_history = []
        fired = False
        wait_count = 0

        print("\n🚀 开始测试：悬停等待开火...\n")

        while step < self.step_limit:
            uav_pos = uav.position
            dist_to_weapon = compute_distance(self.weapon_pos, uav_pos)
            dist_history.append(dist_to_weapon)

            # 获取武器状态
            weapon_state = self.get_weapon_state()
            state_name = self._get_weapon_state_name(weapon_state)
            bullet_count = self.get_bullet_count()

            # 判断是否已进入开火状态
            if weapon_state == 3:
                fired = True
                print(f"🔥 检测到开火状态！(步数 {step})，切换为圆周运动")

            if not fired:
                # 未开火：悬停等待
                action_vec = self.compute_hold_action(uav_pos)
                wait_count += 1
                if wait_count > self.wait_steps:
                    print(f"⚠️ 等待 {self.wait_steps} 步仍未开火，强制开始圆周运动（可能武器参数问题）")
                    fired = True  # 强制进入圆周运动，以免死循环
            else:
                # 已开火：圆周运动
                action_vec = self.compute_circular_action(uav_pos, step)

            action_list = [action_vec.tolist()]
            if self.num_agents > 1:
                action_list.append([0.0, 0.0, 0.0])

            obs, rewards, done, info = self.env.step(action_list)

            uav = self.env.raw_uavs[0]
            if uav.status != UAVState.ALIVE:
                print(f"\n💥 无人机在第 {step} 步被击毁！")
                hit = True
                break

            if True:
                phase = "悬停" if not fired else "圆周"
                print(f"[{step:3d}] {phase} | 状态: {state_name} | 距离: {dist_to_weapon:.1f}m | 子弹: {bullet_count}")

            state_history.append(weapon_state)
            step += 1

            # 如果距离偏离半径太多，可能脱离开火区，提前结束
            # if fired and abs(dist_to_weapon - self.radius) > 100:
            #     print(f"\n⚠️ 距离偏离半径过大 ({dist_to_weapon:.1f}m)，测试结束")
            #     break

            # 如果超时未开火，强制结束
            if not fired and wait_count > self.wait_steps + 10:
                print(f"\n⚠️ 长时间未进入开火状态，测试结束")
                break

        if not hit and step < self.step_limit:
            print(f"\n✅ 无人机存活了 {step} 步（达到最大步数或主动停止）")

        self.print_summary(state_history, hit, step, dist_history, fired)

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

    def print_summary(self, state_history, hit, steps, dist_history, fired):
        print("\n" + "=" * 70)
        print("📊 测试总结")
        print("=" * 70)
        print(f"是否进入开火状态: {'是' if fired else '否'}")
        print(f"总步数: {steps}")
        print(f"被击毁: {'是' if hit else '否'}")
        print(f"初始距离: {dist_history[0]:.1f}m, 最终距离: {dist_history[-1]:.1f}m")
        print("武器状态分布:")
        states = state_history
        state_counts = {s: states.count(s) for s in set(states)}
        for s, count in sorted(state_counts.items()):
            name = self._get_weapon_state_name(s)
            print(f"  {name}: {count} 步")
        print("=" * 70)


def main():
    tester = EscapeAfterFireTester(
        speed=1000050.0,       # 切向速度 (m/s)
        radius=1500.0,     # 圆周半径 (m)
        step_limit=300,
        wait_steps=50,     # 最多等待50步（5秒）开火
        start_dist=1450    # 初始距离
    )
    tester.run()


if __name__ == "__main__":
    main()