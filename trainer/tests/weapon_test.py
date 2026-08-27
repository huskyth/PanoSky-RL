"""
武器系统测试脚本 - 直接飞向目标
验证：搜索雷达标记、追踪雷达调弦/捕获、开火、子弹发射
"""

import numpy as np
import time
import configparser
import json
from pathlib import Path

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
        self.log_data = []
        self.env = None
        self.weapon_state_names = ['NORMAL', 'TUNING', 'CAPTURE', 'FIRE', 'RELOAD']

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
            is_debug=False,          # 减少日志输出
            is_share=True,
            is_use_weapon=True
        )

        obs = self.env.reset()
        self.target = np.array(self.env.target)
        self.weapon_pos = np.array(self.env.weapon)
        self.num_agents = self.env.n_total_uavs

        print("=" * 70)
        print("🎯 武器系统测试开始")
        print("=" * 70)
        print(f"📌 目标位置: {self.target}")
        print(f"🔫 武器位置: {self.weapon_pos}")
        print(f"🚁 无人机: {self.num_agents} 架 (诱饵={self.env.n_decoy_uavs}, 任务机={self.env.n_task_uavs})")
        for i, uav in enumerate(self.env.raw_uavs):
            print(f"   UAV-{i}: 位置={uav.position}, 状态={self._get_uav_status_name(uav.status)}")
        print(f"⚡ 飞行速度: {self.speed} m/s (最大: {self.env.uav_velocity_value} m/s)")
        print("=" * 70)
        return obs

    def compute_action(self, uav_pos):
        """计算指向目标的动作向量（归一化速度指令）"""
        direction = self.target - np.array(uav_pos)
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0, 0.0])
        # 速度指令 = 期望速度 / 最大速度，截断到 [-1, 1]
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

        # 目标无人机
        uav = self.env.raw_uavs[self.uav_idx]
        # 让另一架无人机悬停在地图角落（远离目标区域）
        if self.num_agents > 1:
            for i in range(self.num_agents):
                if i != self.uav_idx:
                    self.env.raw_uavs[i].position = [self.env.map.map_max_x - 100,
                                                     self.env.map.map_max_y - 100,
                                                     self.env.raw_uavs[i].position[2]]
                    self.env.raw_uavs[i].velocity = [0, 0, 0]

        # 初始动作
        action_vec = self.compute_action(uav.position)
        actions = []
        for i in range(self.num_agents):
            if i == self.uav_idx:
                actions.append(action_vec.tolist())
            else:
                actions.append([0.0, 0.0, 0.0])

        step = 0
        last_state = -1
        weapon_state_history = []

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

            # 获取攻击状态（注意属性名可能是 attacked_state）
            attack_state = getattr(uav, 'attacked_state', AttackState.SAFE)
            attack_str = attack_state.name if hasattr(attack_state, 'name') else str(attack_state)

            # 记录
            self.log_data.append({
                'step': step,
                'dist_to_target': float(dist_to_target),
                'dist_to_weapon': float(dist_to_weapon),
                'weapon_state': weapon_state,
                'is_locked': is_locked,
                'bullet_count': bullet_count,
                'uav_status': self._get_uav_status_name(uav.status),
                'attack_state': attack_str,
                'reward': rewards[self.uav_idx]
            })

            # 打印日志
            if step % self.log_interval == 0 or weapon_state != last_state:
                print(f"[{step:3d}] 目标距离: {dist_to_target:6.1f}m | "
                      f"武器距离: {dist_to_weapon:6.1f}m | "
                      f"状态: {state_name:>8} | "
                      f"锁定: {'✅' if is_locked else '❌'} | "
                      f"子弹: {bullet_count:2d} | "
                      f"UAV: {self._get_uav_status_name(uav.status)} | "
                      f"奖励: {rewards[self.uav_idx]:6.3f}")
                last_state = weapon_state

            weapon_state_history.append(weapon_state)

            # 判断是否被击毁（根据状态或攻击状态）
            if uav.status == UAVState.DESTROYED or attack_state == AttackState.DESTROYED:
                print(f"💥 无人机 {self.uav_idx} 被击毁！")
                break

            # 如果进入开火状态且距离很近，但状态还没变，我们可以模拟一次命中（可选）
            # 这里不模拟，让真实逻辑触发。

            # 到达目标
            if dist_to_target <= self.env.task_success_radius:
                print(f"✅ 无人机到达目标！距离: {dist_to_target:.1f}m")
                break

            step += 1

        self.print_summary(weapon_state_history)
        self.save_log()
        self.env.close()
        return self.log_data

    def print_summary(self, state_history):
        print("\n" + "=" * 70)
        print("📊 测试总结")
        print("=" * 70)
        uav = self.env.raw_uavs[self.uav_idx]
        uav_pos = np.array(uav.position)
        dist_to_target = compute_distance(uav_pos, self.target)

        bullet_events = [d for d in self.log_data if d['bullet_count'] > 0]
        print(f"总步数: {len(self.log_data)}")
        print(f"无人机状态: {self._get_uav_status_name(uav.status)}")
        print(f"最终距目标: {dist_to_target:.1f}m")
        print(f"子弹发射次数: {len(bullet_events)}")
        if bullet_events:
            print(f"首次发射: 第 {bullet_events[0]['step']} 步")

        # 分析状态转换
        states = [d['weapon_state'] for d in self.log_data]
        state_counts = {s: states.count(s) for s in set(states)}
        print("\n武器状态分布:")
        for s, count in sorted(state_counts.items()):
            name = self._get_weapon_state_name(s)
            print(f"  {name}: {count} 步")

        # 检查流程
        if 0 in states and 1 in states and 2 in states:
            print("\n✅ 武器流程完整: NORMAL → TUNING → CAPTURE")
            if 3 in states:
                print("✅ 进入 FIRE 状态（开火）")
            else:
                print("⚠️ 未进入 FIRE 状态（可能需要更靠近或更长时间）")
        else:
            print("\n⚠️ 武器流程不完整，检查无人机是否进入射程")

        # 检查锁定
        locked_steps = sum(1 for d in self.log_data if d['is_locked'])
        print(f"\n锁定状态: 被锁定 {locked_steps} 步 (共 {len(self.log_data)} 步)")
        print("=" * 70)

    def save_log(self):
        log_file = Path("weapon_test_log.json")
        with open(log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        print(f"📁 日志已保存到 {log_file}")


def main():
    # 可调参数
    tester = WeaponTester(
        speed=100.0,           # 飞行速度 (m/s)
        step_limit=500,
        use_uav_index=0,
        log_interval=10
    )
    tester.run()


if __name__ == "__main__":
    main()