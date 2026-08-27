"""
推理脚本 - 加载训练好的模型，在环境中运行一个 Episode 并保存 JSON 数据
用于可视化前端查看突防轨迹
"""

import os
import sys
import time
import argparse
import configparser
import numpy as np
import torch
from pathlib import Path

# 导入环境和网络
from trainer.drone.mul_uav_env import MultiUavEnv
from trainer.train import Actor  # 复用训练脚本中的 Actor 定义


def load_actor(checkpoint_path, obs_dim, action_dim, device):
    """加载模型权重"""
    actor = Actor(obs_dim, action_dim).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def inference(args):
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载配置
    config_path = f'../drone/config/{args.config}'
    cf = configparser.ConfigParser()
    cf.read(config_path, encoding='utf-8')

    # 创建环境（必须开启 is_debug=True 才能记录 JSON）
    env = MultiUavEnv(
        rank=0,
        mode="test",
        cf=cf,
        episode_limit=args.episode_limit,
        is_debug=True,          # 关键：启用数据记录
        is_share=True,
        is_use_weapon=args.use_weapon
    )

    # 获取维度
    obs_list = env.reset()
    obs_dim = len(obs_list[0])
    action_dim = env.action_space[0].shape[0]
    num_agents = env.n_total_uavs
    print(f"观测维度: {obs_dim}, 动作维度: {action_dim}, 智能体数: {num_agents}")

    # 加载模型
    if args.checkpoint is None:
        # 自动查找最新的 checkpoint
        ckpt_dir = Path("../checkpoints")
        if not ckpt_dir.exists():
            raise FileNotFoundError("checkpoints 目录不存在，请指定 --checkpoint")
        checkpoints = sorted(ckpt_dir.glob("mappo_actor_step_*.pth"), key=lambda p: int(p.stem.split('_')[-1]))
        if not checkpoints:
            raise FileNotFoundError("未找到任何 checkpoint 文件")
        checkpoint_path = checkpoints[-1]
        print(f"自动使用最新 checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint

    actor = load_actor(checkpoint_path, obs_dim, action_dim, device)

    # 执行推理
    obs_list = env.reset()
    obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)

    step = 0
    total_reward = 0
    done = False

    print("\n🚀 开始推理...")
    while not done and step < args.episode_limit:
        with torch.no_grad():
            action_dist = actor(obs_tensor)
            if args.deterministic:
                action = action_dist.mean  # 确定性策略（均值）
            else:
                action = action_dist.sample()  # 随机采样
            action_np = torch.clamp(action, -1.0, 1.0).cpu().numpy()
            action_list = [action_np[i] for i in range(num_agents)]

        next_obs_list, rewards_list, term_list, info = env.step(action_list)

        total_reward += sum(rewards_list)
        done = all(term_list)   # 根据您的环境逻辑，全部阵亡才算结束
        step += 1

        obs_tensor = torch.tensor(np.array(next_obs_list), dtype=torch.float32).to(device)

        # 打印进度
        if step % 10 == 0:
            dist_to_target0 = np.linalg.norm(np.array(env.raw_uavs[0].position) - np.array(env.target))
            dist_to_target1 = np.linalg.norm(np.array(env.raw_uavs[1].position) - np.array(env.target))
            print(f"Step {step}: UAV0 距目标 {dist_to_target0:.1f}m, UAV1 距目标 {dist_to_target1:.1f}m")

    # 结束
    print(f"\n✅ 推理完成，共 {step} 步，总奖励: {total_reward:.2f}")
    print(f"📁 数据文件已保存到: {env.episode_data_file}")
    print("💡 用前端 HTML 加载此 JSON 文件查看轨迹")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="无人机突防推理脚本")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型权重路径，默认自动使用最新的 checkpoint")
    parser.add_argument("--config", type=str, default="th_demo.ini", help="配置文件名称")
    parser.add_argument("--episode_limit", type=int, default=500, help="最大步数")
    parser.add_argument("--use_weapon", action="store_true", default=True, help="是否开启武器")
    parser.add_argument("--deterministic", action="store_true", default=False, help="使用确定性动作（均值）")
    args = parser.parse_args()

    inference(args)