import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import gymnasium as gym

# ==========================================
# 修改：导入你自己的环境
# ==========================================
# 假设你的 MultiUavEnv 在 onpolicy.envs.drone 包下
# 根据你的实际路径修改这行
from onpolicy.envs.drone.multi_uav_env import MultiUavEnv
from onpolicy.envs.drone.utils.config_loader import load_config  # 如果你有配置加载器


# 或者直接传入 cf 对象


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


# ==========================================
# 2. 神经网络结构（保持不变）
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
# 3. 经验回放池 (Rollout Buffer) - 适配你的环境
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
# 4. 训练主循环（适配 MultiUavEnv）
# ==========================================
def train():
    args = PPOArgs()

    # ==========================================
    # 修改：创建你自己的环境实例
    # ==========================================
    # 方式一：如果有配置文件
    # cf = load_config("path/to/config.ini")
    # env = MultiUavEnv(rank=0, mode="train", cf=cf, episode_limit=500, is_debug=False, is_share=True)

    # 方式二：直接传入 cf 参数（需要先构造）
    # 如果你暂时没有配置文件，可以传 None，然后在 init_from_config 里用默认值
    env = MultiUavEnv(
        rank=0,
        mode="train",
        cf=None,  # 如果你有配置文件，替换这里
        episode_limit=500,
        is_debug=False,
        is_share=True
    )

    # ==========================================
    # 获取维度（适配你的环境）
    # ==========================================
    # 注意：你的环境返回的是 list of obs，而不是 dict
    # 先用 reset 获取初始观测
    obs_list = env.reset()  # obs_list 是 list of np.array

    # 单智能体观测维度
    # 你的 obs 是 28 维向量（在 get_observation_of_a_uav 里定义的）
    if env.is_share:
        # 如果是 share 模式，obs 是一个 dict，有 "linear" 字段
        obs_dim = env.observation_space[0]["linear"].shape[0]
        # 或者直接取实际 obs 的维度
        # obs_dim = len(obs_list[0])
    else:
        # 如果是非 share 模式，obs 直接是 Box 向量
        obs_dim = env.observation_space[0].shape[0]

    # 动作维度：你的动作是 Box(3)
    action_dim = env.action_space[0].shape[0]

    # 全局状态维度 = 所有智能体的观测拼接
    num_agents = env.n_total_uavs  # 从环境获取智能体数量
    global_state_dim = obs_dim * num_agents

    print(f"obs_dim: {obs_dim}, action_dim: {action_dim}, num_agents: {num_agents}")

    # 实例化网络与优化器
    actor = Actor(obs_dim, action_dim).to(args.device)
    critic = Critic(global_state_dim).to(args.device)
    optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': args.lr},
        {'params': critic.parameters(), 'lr': args.lr}
    ], eps=1e-5)

    buffer = RolloutBuffer(args.num_steps, num_agents, obs_dim, global_state_dim, action_dim, args.device)

    # 日志探针数据结构
    ep_rewards = deque(maxlen=args.log_interval)
    ep_lengths = deque(maxlen=args.log_interval)
    ep_success = deque(maxlen=args.log_interval)  # 记录成功率
    all_episode_rewards = []

    global_step = 0
    start_time = time.time()

    # ==========================================
    # 修改：处理初始观测（适配你的环境）
    # ==========================================
    obs_list = env.reset()
    # 转换为 tensor [num_agents, obs_dim]
    obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).to(args.device)

    current_ep_reward = 0
    current_ep_length = 0
    current_ep_success = 0  # 1 表示成功，0 表示失败

    print(f"[{time.strftime('%H:%M:%S')}] 开始 MAPPO 训练 | 设备: {args.device} | 目标步数: {args.total_timesteps}")
    print(f"智能体数量: {num_agents}, 观测维度: {obs_dim}, 动作维度: {action_dim}")
    print("-" * 80)
    print(f"{'Step':>10} | {'Ep Reward':>10} | {'Ep Len':>8} | {'Success Rate':>10} | {'FPS':>6}")
    print("-" * 80)

    while global_step < args.total_timesteps:
        # -------------------------------------
        # 阶段 A：收集数据 (Rollout)
        # -------------------------------------
        actor.eval()
        critic.eval()

        for step in range(args.num_steps):
            global_step += 1

            # 构建全局状态
            global_state_tensor = obs_tensor.flatten().unsqueeze(0)  # [1, global_state_dim]

            with torch.no_grad():
                # Actor 给出动作（所有智能体共享同一个策略网络）
                action_dist = actor(obs_tensor)  # obs_tensor: [num_agents, obs_dim]
                action = action_dist.sample()  # [num_agents, action_dim]
                logprob = action_dist.log_prob(action).sum(1)  # [num_agents]

                # Critic 给出价值（输入全局状态）
                value = critic(global_state_tensor).flatten()  # [1] -> 标量
                value_expanded = value.expand(num_agents)  # [num_agents]

            # 裁剪动作到 [-1, 1]
            action_np = torch.clamp(action, -1.0, 1.0).cpu().numpy()

            # 你的环境 step 接收的是 list of actions
            # 每个 action 是 shape=(3,) 的 numpy array
            action_list = [action_np[i] for i in range(num_agents)]

            # ==========================================
            # 执行步进
            # ==========================================
            next_obs_list, rewards_list, term_list, info = env.step(action_list)
            # 注意：你的环境返回的 term_list 是 list of bool

            # 转换数据格式
            reward_tensor = torch.tensor(np.array(rewards_list).flatten(), dtype=torch.float32).to(args.device)
            done_tensor = torch.tensor(np.array(term_list), dtype=torch.float32).to(args.device)
            next_obs_tensor = torch.tensor(np.array(next_obs_list), dtype=torch.float32).to(args.device)

            # 存入 Buffer
            buffer.add(obs_tensor, global_state_tensor.squeeze(0), action, logprob,
                       reward_tensor, done_tensor, value_expanded)

            obs_tensor = next_obs_tensor
            current_ep_reward += np.mean(rewards_list)
            current_ep_length += 1

            # 判断是否成功（你的环境里，任务成功时 term_list 全为 True，且 rewards 里有 +100）
            if any(term_list):
                # 检查是否有任务成功奖励
                if max(rewards_list) > 50:  # 有 +100 奖励，说明任务成功
                    current_ep_success = 1
                else:
                    current_ep_success = 0

            # 回合结束逻辑
            if any(term_list):
                ep_rewards.append(current_ep_reward)
                ep_lengths.append(current_ep_length)
                ep_success.append(current_ep_success)

                all_episode_rewards.append(current_ep_reward)

                # 打印探针日志
                if len(ep_rewards) >= args.log_interval:
                    fps = int(global_step / (time.time() - start_time))
                    success_rate = np.mean(ep_success) * 100
                    print(f"{global_step:>10} | "
                          f"{np.mean(ep_rewards):>10.2f} | "
                          f"{np.mean(ep_lengths):>8.0f} | "
                          f"{success_rate:>9.1f}% | "
                          f"{fps:>6}")
                    # 清空队列
                    ep_rewards.clear()
                    ep_lengths.clear()
                    ep_success.clear()

                # 重置环境
                obs_list = env.reset()
                obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).to(args.device)
                current_ep_reward = 0
                current_ep_length = 0
                current_ep_success = 0

        # -------------------------------------
        # 阶段 B：GAE 计算与 PPO 更新
        # -------------------------------------
        actor.train()
        critic.train()

        # 获取最后一步的价值用于 Bootstrap
        global_state_tensor = obs_tensor.flatten().unsqueeze(0)
        with torch.no_grad():
            next_value = critic(global_state_tensor).flatten().expand(num_agents)

        # 计算 GAE 和 Returns
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

        # 展平 Buffer 数据
        b_obs = buffer.obs.view(-1, obs_dim)
        b_global_states = buffer.global_states.unsqueeze(1).expand(-1, num_agents, -1).reshape(-1, global_state_dim)
        b_actions = buffer.actions.view(-1, action_dim)
        b_logprobs = buffer.logprobs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = buffer.values.view(-1)

        # 学习率退火
        frac = 1.0 - (global_step - 1.0) / args.total_timesteps
        current_lr = args.lr * frac
        optimizer.param_groups[0]["lr"] = current_lr
        optimizer.param_groups[1]["lr"] = current_lr

        b_inds = np.arange(args.num_steps * num_agents)
        for epoch in range(args.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.num_steps * num_agents, args.batch_size):
                end = start + args.batch_size
                mb_inds = b_inds[start:end]

                # Actor 损失计算
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

                # Critic 损失计算
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

        buffer.clear()

        # -------------------------------------
        # 阶段 C：模型持久化
        # -------------------------------------
        if global_step % args.save_interval == 0 or global_step >= args.total_timesteps:
            os.makedirs("checkpoints", exist_ok=True)
            model_path = f"checkpoints/mappo_actor_step_{global_step}.pth"
            torch.save(actor.state_dict(), model_path)
            print(f">>> 模型已保存至: {model_path}")

    # ==========================================
    # 5. 训练结束
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 训练结束！共耗时 {(time.time() - start_time) / 3600:.2f} 小时。")
    print(f"总回合数: {len(all_episode_rewards)}")
    print(f"平均奖励: {np.mean(all_episode_rewards):.2f}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    train()