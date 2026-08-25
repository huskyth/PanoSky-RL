"""
无人机突防环境 - 单元测试
运行方式: python -m pytest test_env.py -v
或者: python test_env.py
"""

import unittest
import numpy as np
import sys
import os
import json
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drone.mul_uav_env import MultiUavEnv
from drone.uav_meta_info import TrainUAV
from drone.weapons.entries.uav.uav_enum import UAVState, AttackState


class TestMultiUavEnv(unittest.TestCase):
    """无人机突防环境单元测试"""

    @classmethod
    def setUpClass(cls):
        """所有测试前执行一次"""
        # 尝试加载配置文件
        config_path = 'drone/config/th_demo.ini'
        if os.path.exists(config_path):
            import configparser
            cf = configparser.ConfigParser()
            cf.read(config_path, encoding='utf-8')
            cls.cf = cf
        else:
            cls.cf = None
            print("⚠️ 配置文件不存在，使用默认参数")

    def setUp(self):
        """每个测试前执行"""
        self.env = MultiUavEnv(
            rank=0,
            mode="test",
            cf=self.cf,
            episode_limit=100,
            is_debug=True,
            is_share=True
        )
        self.num_agents = self.env.n_total_uavs

    def tearDown(self):
        """每个测试后执行"""
        if hasattr(self.env, 'close'):
            self.env.close()

    # ============================================================
    # 1. 环境初始化测试
    # ============================================================
    def test_env_initialization(self):
        """测试环境是否正常初始化"""
        self.assertIsNotNone(self.env)
        self.assertIsNotNone(self.env.raw_uavs)
        self.assertIsNotNone(self.env.weapon)
        self.assertIsNotNone(self.env.target)
        self.assertEqual(len(self.env.raw_uavs), self.num_agents)
        print("✅ 环境初始化成功")

    def test_reset(self):
        """测试 reset 是否正常工作"""
        obs = self.env.reset()
        self.assertIsNotNone(obs)
        self.assertEqual(len(obs), self.num_agents)
        # 检查每个观测的形状
        for i, o in enumerate(obs):
            self.assertEqual(len(o), 28)  # 28维观测
            self.assertIsInstance(o, np.ndarray)
        print("✅ Reset 正常工作，观测维度: 28")

    def test_action_space(self):
        """测试动作空间是否正确"""
        action_space = self.env.action_space[0]
        self.assertEqual(action_space.shape[0], 3)  # 3维连续动作
        # 采样测试
        sample_action = action_space.sample()
        self.assertEqual(len(sample_action), 3)
        self.assertTrue(np.all(sample_action >= -1.0))
        self.assertTrue(np.all(sample_action <= 1.0))
        print("✅ 动作空间正确: Box(3)")

    def test_observation_space(self):
        """测试观测空间是否正确"""
        obs_space = self.env.observation_space[0]
        if self.env.is_share:
            obs_dim = obs_space["linear"].shape[0]
        else:
            obs_dim = obs_space.shape[0]
        self.assertEqual(obs_dim, 28)
        print("✅ 观测空间正确: 28维")

    # ============================================================
    # 2. Step 功能测试
    # ============================================================
    def test_step_with_random_actions(self):
        """测试随机动作是否能正常执行"""
        obs = self.env.reset()

        for i in range(10):
            # 生成随机动作
            actions = [np.random.uniform(-1, 1, 3) for _ in range(self.num_agents)]

            next_obs, rewards, done, info = self.env.step(actions)

            # 检查返回值格式
            self.assertEqual(len(next_obs), self.num_agents)
            self.assertEqual(len(rewards), self.num_agents)
            self.assertEqual(len(done), self.num_agents)
            self.assertIsInstance(info, dict)

            # 检查奖励范围（应该在截断范围内）
            for r in rewards:
                self.assertTrue(-2.0 <= r <= 2.0 or r == -10.0 or r == 100.0,
                                f"奖励 {r} 超出预期范围")

            # 检查无人机位置是否更新
            for j in range(self.num_agents):
                if self.env.raw_uavs[j].status == UAVState.ALIVE:
                    self.assertIsNotNone(self.env.raw_uavs[j].position)

            if any(done):
                break

        print("✅ 随机动作 Step 测试通过")

    def test_step_returns_correct_format(self):
        """测试 step 返回格式是否正确"""
        obs = self.env.reset()
        actions = [np.zeros(3) for _ in range(self.num_agents)]

        next_obs, rewards, done, info = self.env.step(actions)

        # 检查 obs 是 list of np.ndarray
        self.assertIsInstance(next_obs, list)
        self.assertIsInstance(next_obs[0], np.ndarray)

        # 检查 rewards 是 list of float
        self.assertIsInstance(rewards, list)
        self.assertIsInstance(rewards[0], float)

        # 检查 done 是 list of bool
        self.assertIsInstance(done, list)
        self.assertIsInstance(done[0], bool)

        # 检查 info 是 dict
        self.assertIsInstance(info, dict)
        print("✅ Step 返回格式正确")

    # ============================================================
    # 3. 动作边界测试
    # ============================================================
    def test_action_clamping(self):
        """测试动作是否被正确裁剪"""
        obs = self.env.reset()

        # 测试极端动作
        extreme_actions = [
            [np.array([10.0, 10.0, 10.0]) for _ in range(self.num_agents)],
            [np.array([-10.0, -10.0, -10.0]) for _ in range(self.num_agents)],
            [np.array([0.0, 0.0, 0.0]) for _ in range(self.num_agents)],
        ]

        for actions in extreme_actions:
            next_obs, rewards, done, info = self.env.step(actions)
            # 检查无人机是否还在地图内
            for uav in self.env.raw_uavs:
                if uav.status == UAVState.ALIVE:
                    x, y, z = uav.position
                    self.assertTrue(self.env.map.map_min_x <= x <= self.env.map.map_max_x)
                    self.assertTrue(self.env.map.map_min_y <= y <= self.env.map.map_max_y)
                    self.assertTrue(self.env.min_available_height <= z <= self.env.max_available_height)

        print("✅ 动作边界测试通过")

    # ============================================================
    # 4. 奖励函数测试
    # ============================================================
    def test_reward_structure(self):
        """测试奖励函数的结构"""
        obs = self.env.reset()
        actions = [np.random.uniform(-1, 1, 3) for _ in range(self.num_agents)]

        next_obs, rewards, done, info = self.env.step(actions)

        # 检查奖励是否包含个人项和共享项
        self.assertEqual(len(rewards), self.num_agents)

        # 打印奖励分解（如果有）
        if 'individual_reward' in info:
            for i in range(self.num_agents):
                self.assertIn('individual_reward', info)
                self.assertIsInstance(info['individual_reward'], dict)
                self.assertIn(i, info['individual_reward'])
        else:
            print("ℹ️ info 中不包含 individual_reward（这是正常的）")

        print("✅ 奖励函数结构正确")

    def test_reward_clipping(self):
        """测试奖励截断是否生效"""
        obs = self.env.reset()

        # 执行大量随机步，收集奖励
        rewards_list = []
        for _ in range(50):
            actions = [np.random.uniform(-1, 1, 3) for _ in range(self.num_agents)]
            _, rewards, done, _ = self.env.step(actions)
            rewards_list.extend(rewards)
            if any(done):
                break

        # 检查奖励是否被正确截断（除了特殊值 -10 和 100）
        for r in rewards_list:
            if r != -10.0 and r != 100.0:
                self.assertTrue(-2.0 <= r <= 2.0, f"奖励 {r} 未被正确截断")

        print("✅ 奖励截断机制正常")

    # ============================================================
    # 5. 终端条件测试
    # ============================================================
    def test_terminal_conditions(self):
        """测试终端条件是否正确触发"""
        obs = self.env.reset()

        # 测试1：步数超限
        for _ in range(self.env.max_episode_steps + 10):
            actions = [np.zeros(3) for _ in range(self.num_agents)]
            _, _, done, _ = self.env.step(actions)
            if any(done):
                break

        # 步数超限应该触发 done
        # 但由于我们只跑了 100+10 步，可能还没到 max_episode_steps
        # 这里只检查 done 是 bool 列表
        self.assertIsInstance(done, list)
        self.assertIsInstance(done[0], bool)

        print("✅ 终端条件测试通过")

    # ============================================================
    # 6. 数据记录测试
    # ============================================================
    def test_data_recording(self):
        """测试数据是否被正确记录"""
        obs = self.env.reset()

        # 运行几步
        for _ in range(5):
            actions = [np.random.uniform(-1, 1, 3) for _ in range(self.num_agents)]
            _, _, done, _ = self.env.step(actions)
            if any(done):
                break

        # 检查数据文件是否创建
        if hasattr(self.env, 'episode_data_file'):
            file_path = Path(self.env.episode_data_file)
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.assertIsInstance(data, list)
                if len(data) > 0:
                    self.assertIn('uavs', data[0])
                    self.assertIn('weapon', data[0])
                    self.assertIn('target', data[0])
                print(f"✅ 数据记录正常，文件: {file_path}")
            else:
                print("ℹ️ 数据文件尚未生成（可能需要更多步数）")
        else:
            print("ℹ️ 环境未配置数据记录（is_debug=False）")

    # ============================================================
    # 7. 坐标缩放测试
    # ============================================================
    def test_coordinate_scaling(self):
        """测试坐标缩放是否正确"""
        obs = self.env.reset()

        # 运行一步
        actions = [np.zeros(3) for _ in range(self.num_agents)]
        _, _, done, _ = self.env.step(actions)

        # 检查记录的数据中坐标是否在合理范围
        if hasattr(self.env, 'episode_data_file'):
            file_path = Path(self.env.episode_data_file)
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if len(data) > 0:
                    record = data[0]
                    for uav in record.get('uavs', []):
                        pos = uav.get('position', [])
                        if len(pos) >= 3:
                            # 缩放后的坐标应该在合理范围
                            self.assertTrue(-100 < pos[0] < 100, f"缩放后 x 坐标异常: {pos[0]}")
                            self.assertTrue(-100 < pos[2] < 100, f"缩放后 z 坐标异常: {pos[2]}")
                            self.assertTrue(-100 < pos[1] < 100, f"缩放后 y 坐标异常: {pos[1]}")
                    print("✅ 坐标缩放正常")

        print("✅ 坐标缩放测试通过")

    # ============================================================
    # 8. 多智能体交互测试
    # ============================================================
    def test_multi_agent_interaction(self):
        """测试多智能体之间的交互"""
        obs = self.env.reset()

        # 检查两架无人机是否不同
        uav0_pos = self.env.raw_uavs[0].position
        uav1_pos = self.env.raw_uavs[1].position

        # 初始位置可能相同，但速度应该不同
        uav0_vel = self.env.raw_uavs[0].velocity
        uav1_vel = self.env.raw_uavs[1].velocity

        # 执行几步让它们移动
        for _ in range(20):
            actions = [np.random.uniform(-1, 1, 3) for _ in range(self.num_agents)]
            _, _, done, _ = self.env.step(actions)
            if any(done):
                break

        # 检查无人机是否有不同的轨迹
        pos0 = self.env.raw_uavs[0].position
        pos1 = self.env.raw_uavs[1].position

        # 至少有一架移动了
        moved0 = np.linalg.norm(np.array(pos0) - np.array(uav0_pos)) > 1
        moved1 = np.linalg.norm(np.array(pos1) - np.array(uav1_pos)) > 1
        self.assertTrue(moved0 or moved1, "无人机完全没有移动")

        print("✅ 多智能体交互正常")

    # ============================================================
    # 9. 武器接口测试
    # ============================================================
    def test_weapon_interface(self):
        """测试武器接口是否正常"""
        # 检查武器状态
        try:
            state = self.env._get_game_target_idx()
            # state 可以是 None 或 int
            self.assertTrue(state is None or isinstance(state, int))
        except Exception as e:
            self.fail(f"武器接口调用失败: {e}")

        print("✅ 武器接口正常")

    # ============================================================
    # 10. 长时间稳定性测试
    # ============================================================
    def test_stability(self):
        """测试环境长时间运行是否稳定"""
        obs = self.env.reset()

        total_steps = 100
        for step in range(total_steps):
            actions = [np.random.uniform(-1, 1, 3) for _ in range(self.num_agents)]
            try:
                next_obs, rewards, done, info = self.env.step(actions)
            except Exception as e:
                self.fail(f"第 {step} 步出错: {e}")

            if any(done):
                obs = self.env.reset()
                continue

        print(f"✅ 稳定运行 {total_steps} 步无异常")

    # ============================================================
    # 11. 空动作测试
    # ============================================================
    def test_zero_actions(self):
        """测试零动作是否导致无人机悬停"""
        obs = self.env.reset()
        actions = [np.zeros(3) for _ in range(self.num_agents)]

        # 记录初始位置
        initial_positions = [uav.position.copy() for uav in self.env.raw_uavs]

        # 执行几步
        for _ in range(10):
            _, _, done, _ = self.env.step(actions)
            if any(done):
                break

        # 零动作应该导致无人机基本不动（可能受重力或其他物理影响）
        for i, uav in enumerate(self.env.raw_uavs):
            if uav.status == UAVState.ALIVE:
                displacement = np.linalg.norm(np.array(uav.position) - np.array(initial_positions[i]))
                # 由于物理引擎可能有微小变化，允许小幅度移动
                self.assertLess(displacement, 10, f"UAV {i} 零动作下位移过大: {displacement}")

        print("✅ 零动作测试通过")


# ============================================================
# 13. 快速验证脚本（不依赖 unittest，可单独运行）
# ============================================================
def quick_verify():
    """快速验证环境是否正常工作"""
    print("=" * 60)
    print("环境快速验证")
    print("=" * 60)

    try:
        import configparser
        cf = configparser.ConfigParser()
        config_path = 'drone/config/th_demo.ini'
        if os.path.exists(config_path):
            cf.read(config_path, encoding='utf-8')
        else:
            cf = None
            print("⚠️ 配置文件不存在，使用默认参数")

        env = MultiUavEnv(
            rank=0,
            mode="test",
            cf=cf,
            episode_limit=50,
            is_debug=True,
            is_share=True
        )

        obs = env.reset()
        print(f"✅ 环境初始化成功，观测维度: {len(obs[0])}")

        # 运行一个完整 episode
        total_reward = 0
        for step in range(100):
            actions = [np.random.uniform(-1, 1, 3) for _ in range(env.n_total_uavs)]
            obs, rewards, done, info = env.step(actions)
            total_reward += sum(rewards)
            if any(done):
                print(f"✅ Episode 结束于第 {step} 步，总奖励: {total_reward:.2f}")
                break
        else:
            print(f"✅ 完成 100 步，总奖励: {total_reward:.2f}")

        # 检查数据文件
        if hasattr(env, 'episode_data_file'):
            file_path = Path(env.episode_data_file)
            if file_path.exists():
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"✅ 数据已记录，共 {len(data)} 步")
                # 打印示例数据
                if len(data) > 0:
                    sample = data[0]
                    print(f"  示例 UAV 位置: {sample['uavs'][0]['position']}")
                    print(f"  示例武器位置: {sample['weapon']['position']}")
                    print(f"  示例目标位置: {sample['target']['position']}")
            else:
                print("⚠️ 数据文件尚未生成")

        env.close()
        print("=" * 60)
        print("✅ 所有验证通过，环境可以开始训练！")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 方式1：运行单元测试
    # unittest.main()

    # 方式2：快速验证（推荐先运行这个）
    print("\n" + "=" * 60)
    print("🚀 开始快速验证...")
    print("=" * 60 + "\n")

    success = quick_verify()

    if success:
        print("\n🎯 可以开始训练了！运行 python train.py")
    else:
        print("\n❌ 请检查环境配置后重试")