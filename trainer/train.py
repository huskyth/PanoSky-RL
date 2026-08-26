import configparser
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import gymnasium as gym

# ==========================================
# 导入 SwanLab
# ==========================================
try:
    import swanlab as sw

    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("⚠️ SwanLab 未安装，请运行: pip install swanlab")

# 导入你自己的环境
from drone.mul_uav_env import MultiUavEnv


# ==========================================
# 1. 超参数配置 (PPO Args)
# ==========================================
class PPOArgs:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = 2_000_000
    num_envs = 1
    num_steps = 2048
    batch_size = 256
    n_epochs = 10

    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    save_interval = 500_000
    log_interval = 50

    config_name = 'th_demo.ini'

    # ===== 武器开关 =====
    is_use_weapon = False  # 默认开启武器


# ==========================================
# 2. 神经网络结构
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        action_mean = torch.tanh(self.net(x))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs


class Critic(nn.Module):
    def __init__(self, global_state_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(global_state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. 经验回放池 (Rollout Buffer)
# ==========================================
class RolloutBuffer:
    def __init__(self, num_steps, num_agents, obs_dim, global_state_dim, action_dim, device):
        self.obs = torch.zeros((num_steps, num_agents, obs_dim)).to(device)
        self.global_states = torch.zeros((num_steps, global_state_dim)).to(device)
        self.actions = torch.zeros((num_steps, num_agents, action_dim)).to(device)
        self.logprobs = torch.zeros((num_steps, num_agents)).to(device)
        self.rewards = torch.zeros((num_steps, num_agents)).to(device)
        self.dones = torch.zeros((num_steps, num_agents)).to(device)
        self.values = torch.zeros((num_steps, num_agents)).to(device)

        self.step = 0
        self.device = device

    def add(self, obs, global_state, action, logprob, reward, done, value):
        self.obs[self.step] = obs
        self.global_states[self.step] = global_state
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step += 1

    def clear(self):
        self.step = 0


# ==========================================
# 4. 训练主循环
# ==========================================
def train():
    args = PPOArgs()

    # ==========================================
    # SwanLab 初始化
    # ==========================================
    if SWANLAB_AVAILABLE:
        # 从环境变量读取 API Key（推荐）
        api_key = os.environ.get("SWANLAB_API_KEY", None)
        if api_key:
            sw.login(api_key=api_key)

        config = {
            "total_timesteps": args.total_timesteps,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_coef": args.clip_coef,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "config_name": args.config_name,
            "is_use_weapon": args.is_use_weapon,
        }

        sw.init(
            project="UAV_MAPPO",
            experiment_name=f"mappo_{time.strftime('%Y%m%d_%H%M%S')}",
            config=config,
            log_level="info"
        )
        print("✅ SwanLab 已初始化")
    else:
        print("⚠️ SwanLab 不可用，跳过实验跟踪")

    # ---- 加载配置文件 ----
    config_path = f'drone/config/{args.config_name}'
    cf = configparser.ConfigParser()
    cf.read(str(config_path), encoding="utf-8")

    # ---- 创建环境 ----
    env = MultiUavEnv(
        rank=0,
        mode="train",
        cf=cf,
        episode_limit=500,
        is_debug=False,
        is_share=True,
        is_use_weapon=args.is_use_weapon
    )

    # ---- 获取维度（关键修复：直接取实际观测长度） ----
    obs_list = env.reset()
    obs_dim = len(obs_list[0])  # 直接取实际观测的长度，保证与 env.step 返回一致
    action_dim = env.action_space[0].shape[0]
    num_agents = env.n_total_uavs
    global_state_dim = obs_dim * num_agents

    print(f"obs_dim: {obs_dim}, action_dim: {action_dim}, num_agents: {num_agents}")
    print(f"is_use_weapon: {args.is_use_weapon}")

    # ---- 实例化网络 ----
    actor = Actor(obs_dim, action_dim).to(args.device)
    critic = Critic(global_state_dim).to(args.device)
    optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': args.lr},
        {'params': critic.parameters(), 'lr': args.lr}
    ], eps=1e-5)

    buffer = RolloutBuffer(args.num_steps, num_agents, obs_dim, global_state_dim, action_dim, args.device)

    # ---- 日志 ----
    ep_rewards = deque(maxlen=args.log_interval)
    ep_lengths = deque(maxlen=args.log_interval)
    ep_success = deque(maxlen=args.log_interval)
    all_episode_rewards = []

    global_step = 0
    start_time = time.time()

    obs_list = env.reset()
    obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).to(args.device)

    current_ep_reward = 0
    current_ep_length = 0
    current_ep_success = 0

    print(f"[{time.strftime('%H:%M:%S')}] 开始 MAPPO 训练 | 设备: {args.device} | 目标步数: {args.total_timesteps}")
    print(f"智能体数量: {num_agents}, 观测维度: {obs_dim}, 动作维度: {action_dim}")
    print("-" * 80)
    print(f"{'Step':>10} | {'Ep Reward':>10} | {'Ep Len':>8} | {'Success Rate':>10} | {'FPS':>6}")
    print("-" * 80)

    if SWANLAB_AVAILABLE:
        sw.log({"train/start": 1}, step=0)

    while global_step < args.total_timesteps:
        # ---- 阶段 A：收集数据 ----
        actor.eval()
        critic.eval()

        for step in range(args.num_steps):
            global_step += 1

            global_state_tensor = obs_tensor.flatten().unsqueeze(0)

            with torch.no_grad():
                action_dist = actor(obs_tensor)
                action = action_dist.sample()
                logprob = action_dist.log_prob(action).sum(1)
                value = critic(global_state_tensor).flatten()
                value_expanded = value.expand(num_agents)

            action_np = torch.clamp(action, -1.0, 1.0).cpu().numpy()
            action_list = [action_np[i] for i in range(num_agents)]

            next_obs_list, rewards_list, term_list, info = env.step(action_list)

            reward_tensor = torch.tensor(np.array(rewards_list).flatten(), dtype=torch.float32).to(args.device)
            done_tensor = torch.tensor(np.array(term_list), dtype=torch.float32).to(args.device)
            next_obs_tensor = torch.tensor(np.array(next_obs_list), dtype=torch.float32).to(args.device)

            buffer.add(obs_tensor, global_state_tensor.squeeze(0), action, logprob,
                       reward_tensor, done_tensor, value_expanded)

            obs_tensor = next_obs_tensor
            current_ep_reward += np.mean(rewards_list)
            current_ep_length += 1

            if any(term_list):
                if max(rewards_list) > 50:
                    current_ep_success = 1
                else:
                    current_ep_success = 0

            if any(term_list):
                ep_rewards.append(current_ep_reward)
                ep_lengths.append(current_ep_length)
                ep_success.append(current_ep_success)
                all_episode_rewards.append(current_ep_reward)

                if SWANLAB_AVAILABLE:
                    sw.log({
                        "episode/reward": current_ep_reward,
                        "episode/length": current_ep_length,
                        "episode/success": current_ep_success,
                    }, step=global_step)

                if len(ep_rewards) >= args.log_interval:
                    fps = int(global_step / (time.time() - start_time))
                    success_rate = np.mean(ep_success) * 100
                    avg_reward = np.mean(ep_rewards)
                    avg_length = np.mean(ep_lengths)

                    print(f"{global_step:>10} | "
                          f"{avg_reward:>10.2f} | "
                          f"{avg_length:>8.0f} | "
                          f"{success_rate:>9.1f}% | "
                          f"{fps:>6}")

                    if SWANLAB_AVAILABLE:
                        sw.log({
                            "train/avg_reward": avg_reward,
                            "train/avg_length": avg_length,
                            "train/success_rate": success_rate,
                            "train/fps": fps,
                            "train/global_step": global_step,
                        }, step=global_step)

                    ep_rewards.clear()
                    ep_lengths.clear()
                    ep_success.clear()

                obs_list = env.reset()
                obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).to(args.device)
                current_ep_reward = 0
                current_ep_length = 0
                current_ep_success = 0

        # ---- 阶段 B：GAE 计算与 PPO 更新 ----
        actor.train()
        critic.train()

        global_state_tensor = obs_tensor.flatten().unsqueeze(0)
        with torch.no_grad():
            next_value = critic(global_state_tensor).flatten().expand(num_agents)

        advantages = torch.zeros_like(buffer.rewards).to(args.device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - done_tensor
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - buffer.dones[t + 1]
                nextvalues = buffer.values[t + 1]
            delta = buffer.rewards[t] + args.gamma * nextvalues * nextnonterminal - buffer.values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + buffer.values

        b_obs = buffer.obs.view(-1, obs_dim)
        b_global_states = buffer.global_states.unsqueeze(1).expand(-1, num_agents, -1).reshape(-1, global_state_dim)
        b_actions = buffer.actions.view(-1, action_dim)
        b_logprobs = buffer.logprobs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = buffer.values.view(-1)

        frac = 1.0 - (global_step - 1.0) / args.total_timesteps
        current_lr = args.lr * frac
        optimizer.param_groups[0]["lr"] = current_lr
        optimizer.param_groups[1]["lr"] = current_lr

        b_inds = np.arange(args.num_steps * num_agents)

        epoch_losses = []
        epoch_entropies = []
        epoch_v_losses = []

        for epoch in range(args.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.num_steps * num_agents, args.batch_size):
                end = start + args.batch_size
                mb_inds = b_inds[start:end]

                action_dist = actor(b_obs[mb_inds])
                newlogprob = action_dist.log_prob(b_actions[mb_inds]).sum(1)
                entropy = action_dist.entropy().sum(1).mean()
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = critic(b_global_states[mb_inds]).view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                loss = pg_loss - args.ent_coef * entropy + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_entropies.append(entropy.item())
                epoch_v_losses.append(v_loss.item())

        if SWANLAB_AVAILABLE and epoch_losses:
            sw.log({
                "loss/pg_loss": np.mean(epoch_losses),
                "loss/entropy": np.mean(epoch_entropies),
                "loss/vf_loss": np.mean(epoch_v_losses),
                "loss/learning_rate": current_lr,
            }, step=global_step)

        buffer.clear()

        # ---- 阶段 C：模型持久化 ----
        if global_step % args.save_interval == 0 or global_step >= args.total_timesteps:
            os.makedirs("checkpoints", exist_ok=True)
            model_path = f"checkpoints/mappo_actor_step_{global_step}.pth"
            torch.save(actor.state_dict(), model_path)
            print(f">>> 模型已保存至: {model_path}")

            if SWANLAB_AVAILABLE:
                sw.log({
                    "train/model_saved": 1,
                    "train/save_step": global_step,
                }, step=global_step)
                sw.save(model_path)

    # ==========================================
    # 5. 训练结束
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 训练结束！共耗时 {(time.time() - start_time) / 3600:.2f} 小时。")
    print(f"总回合数: {len(all_episode_rewards)}")
    print(f"平均奖励: {np.mean(all_episode_rewards):.2f}")

    if SWANLAB_AVAILABLE:
        sw.log({
            "final/total_episodes": len(all_episode_rewards),
            "final/avg_reward": np.mean(all_episode_rewards) if all_episode_rewards else 0,
            "final/max_reward": np.max(all_episode_rewards) if all_episode_rewards else 0,
            "final/min_reward": np.min(all_episode_rewards) if all_episode_rewards else 0,
        }, step=global_step)
        sw.finish()
        print("✅ SwanLab 实验已结束")

    # ---- 绘制奖励曲线 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(all_episode_rewards, alpha=0.3, color='blue', label='Raw Reward')

        window = 50
        if len(all_episode_rewards) >= window:
            smoothed = np.convolve(all_episode_rewards, np.ones(window) / window, mode='valid')
            plt.plot(np.arange(window - 1, len(all_episode_rewards)), smoothed,
                     color='red', linewidth=2, label=f'MA ({window})')

        plt.title('MAPPO Training - Average Reward per Episode', fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        os.makedirs("logs", exist_ok=True)
        plt.savefig("logs/reward_curve.png", dpi=150, bbox_inches='tight')
        print("✅ 奖励曲线已保存到 logs/reward_curve.png")

    except ImportError:
        print("⚠️ matplotlib 未安装，请运行: pip install matplotlib")
    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")


if __name__ == "__main__":
    sw.login(api_key="rdGaOSnlBY0KBDnNdkzja")
    torch.set_num_threads(4)
    train()