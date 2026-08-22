import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque

# 导入你的环境与配置
from env import MicroBattleEnv, Config, Plot


# ==========================================
# 1. 超参数配置 (PPO Args)
# ==========================================
class PPOArgs:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = 2_000_000  # 总训练步数
    num_envs = 1
    num_steps = 2048  # 每次更新前收集的步数
    batch_size = 256  # 更新时的 Batch Size
    n_epochs = 10  # 每次收集后，网络优化的轮数

    lr = 3e-4  # 初始学习率
    gamma = 0.99  # 折扣因子
    gae_lambda = 0.95  # GAE 参数
    clip_coef = 0.2  # PPO 截断范围
    ent_coef = 0.01  # 熵奖励系数 (鼓励探索)
    vf_coef = 0.5  # 价值函数损失系数
    max_grad_norm = 0.5  # 梯度裁剪阈值-

    save_interval = 500_000  # 每隔多少步保存一次模型
    log_interval = 50  # 每隔多少个 Episode 打印一次日志


# ==========================================
# 2. 神经网络结构定义
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化权重"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # 纯 MLP 结构，轻量且极易导出为 ONNX 部署
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01)  # 最后一层 std 设小，防止初始动作过大
        )
        # 独立的可学习标准差参数，初始化为 0 (即 exp(0)=1)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        action_mean = torch.tanh(self.net(x))  # 将均值限制在 [-1, 1]
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
    env = MicroBattleEnv()

    # 动态获取维度
    dummy_obs, _ = env.reset()
    obs_dim = env.observation_space["agent_0"].shape[0]
    action_dim = env.action_space["agent_0"].shape[0]
    global_state_dim = obs_dim * Config.NUM_AGENTS  # CTDE 拼接全局状态

    # 实例化网络与优化器
    actor = Actor(obs_dim, action_dim).to(args.device)
    critic = Critic(global_state_dim).to(args.device)
    optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': args.lr},
        {'params': critic.parameters(), 'lr': args.lr}
    ], eps=1e-5)

    buffer = RolloutBuffer(args.num_steps, Config.NUM_AGENTS, obs_dim, global_state_dim, action_dim, args.device)

    # 日志探针数据结构
    ep_rewards = deque(maxlen=args.log_interval)
    ep_lengths = deque(maxlen=args.log_interval)
    ep_boss_hp = deque(maxlen=args.log_interval)
    ep_agents_hp = deque(maxlen=args.log_interval)
    all_episode_rewards = []

    global_step = 0
    start_time = time.time()

    obs_dict, _ = env.reset()
    current_ep_reward = 0
    current_ep_length = 0

    print(f"[{time.strftime('%H:%M:%S')}] 开始 MAPPO 训练 | 设备: {args.device} | 目标步数: {args.total_timesteps}")
    print("-" * 80)
    print(f"{'Step':>10} | {'Ep Reward':>10} | {'Ep Len':>8} | {'Boss HP %':>10} | {'Agents HP %':>11} | {'FPS':>6}")
    print("-" * 80)

    while global_step < args.total_timesteps:
        # -------------------------------------
        # 阶段 A：收集数据 (Rollout)
        # -------------------------------------
        actor.eval()
        critic.eval()

        for step in range(args.num_steps):
            global_step += 1

            # 转换数据格式
            obs_tensor = torch.tensor(np.array(list(obs_dict.values())), dtype=torch.float32).to(args.device)
            global_state_tensor = obs_tensor.flatten().unsqueeze(0)  # [1, 63]

            with torch.no_grad():
                # Actor 给出动作
                action_dist = actor(obs_tensor)
                action = action_dist.sample()
                logprob = action_dist.log_prob(action).sum(1)
                # Critic 给出价值 (由于所有 Agent 共享同一个全局奖励预期，价值函数评估全局状态)
                value = critic(global_state_tensor).flatten()
                # 扩展 value 到各个 Agent 维度
                value_expanded = value.expand(Config.NUM_AGENTS)

            # 裁剪动作并执行步进
            action_np = torch.clamp(action, -1.0, 1.0).cpu().numpy()
            action_dict = {f"agent_{i}": action_np[i] for i in range(Config.NUM_AGENTS)}

            next_obs_dict, rewards_dict, term_dict, trunc_dict, infos = env.step(action_dict)

            # 处理奖励和结束标志
            reward_tensor = torch.tensor(list(rewards_dict.values()), dtype=torch.float32).to(args.device)
            done_tensor = torch.tensor(list(term_dict.values()), dtype=torch.float32).to(args.device)

            buffer.add(obs_tensor, global_state_tensor.squeeze(0), action, logprob, reward_tensor, done_tensor,
                       value_expanded)

            obs_dict = next_obs_dict
            current_ep_reward += sum(rewards_dict.values()) / Config.NUM_AGENTS  # 记录平均单兵总收益
            current_ep_length += 1

            # 回合结束逻辑
            if any(term_dict.values()) or any(trunc_dict.values()):
                ep_rewards.append(current_ep_reward)
                ep_lengths.append(current_ep_length)
                ep_boss_hp.append(infos["env_state"]["boss_hp"] / Config.BOSS_MAX_HP * 100)

                # 计算平均剩余血量
                alive_hps = [env.agents[i, 2] for i in range(Config.NUM_AGENTS) if env.agents[i, 4] == 0]
                avg_agent_hp = sum(alive_hps) / len(alive_hps) if alive_hps else 0.0
                ep_agents_hp.append(avg_agent_hp / Config.AGENT_MAX_HP * 100)

                all_episode_rewards.append(current_ep_reward)

                # 打印探针日志
                if len(ep_rewards) >= args.log_interval and sum(ep_lengths) >= args.log_interval:
                    fps = int(global_step / (time.time() - start_time))
                    print(f"{global_step:>10} | "
                          f"{np.mean(ep_rewards):>10.2f} | "
                          f"{np.mean(ep_lengths):>8.0f} | "
                          f"{np.mean(ep_boss_hp):>9.1f}% | "
                          f"{np.mean(ep_agents_hp):>10.1f}% | "
                          f"{fps:>6}")
                    ep_rewards.clear()  # 清空队列，等下一批

                obs_dict, _ = env.reset()
                current_ep_reward = 0
                current_ep_length = 0

        # -------------------------------------
        # 阶段 B：GAE 计算与 PPO 更新
        # -------------------------------------
        actor.train()
        critic.train()

        # 获取最后一步的价值用于 Bootstrap
        obs_tensor = torch.tensor(np.array(list(obs_dict.values())), dtype=torch.float32).to(args.device)
        global_state_tensor = obs_tensor.flatten().unsqueeze(0)
        with torch.no_grad():
            next_value = critic(global_state_tensor).flatten().expand(Config.NUM_AGENTS)

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
        b_global_states = buffer.global_states.unsqueeze(1).expand(-1, Config.NUM_AGENTS, -1).reshape(-1,
                                                                                                      global_state_dim)
        b_actions = buffer.actions.view(-1, action_dim)
        b_logprobs = buffer.logprobs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = buffer.values.view(-1)

        # 学习率退火 (Linear Decay)
        frac = 1.0 - (global_step - 1.0) / args.total_timesteps
        current_lr = args.lr * frac
        optimizer.param_groups[0]["lr"] = current_lr
        optimizer.param_groups[1]["lr"] = current_lr

        b_inds = np.arange(args.num_steps * Config.NUM_AGENTS)
        for epoch in range(args.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.num_steps * Config.NUM_AGENTS, args.batch_size):
                end = start + args.batch_size
                mb_inds = b_inds[start:end]

                # Actor 损失计算
                action_dist = actor(b_obs[mb_inds])
                newlogprob = action_dist.log_prob(b_actions[mb_inds]).sum(1)
                entropy = action_dist.entropy().sum(1).mean()
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Advantage 归一化 (Batch级别)
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

                # 总损失与反向传播
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
    # 5. 训练结束，绘制曲线
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 训练结束！共耗时 {(time.time() - start_time) / 3600:.2f} 小时。")
    Plot.plot_learning_curve(all_episode_rewards, title="MAPPO Micro Battle - Agent Average Reward", window=100)


if __name__ == "__main__":
    # 为了避免多进程报错，加上这句保护
    torch.set_num_threads(4)
    train()