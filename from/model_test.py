import os
import math
import numpy as np
import torch
import torch.nn as nn
import pygame
import imageio
from collections import deque
from env import MicroBattleEnv, Config


# ==========================================
# 1. 网络架构 (必须与训练时完全一致)
# ==========================================
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        return torch.tanh(self.net(x))  # 评估时使用纯确定性策略


# ==========================================
# 2. 视觉特效与游戏渲染辅助类
# ==========================================
class VisualFX:
    COLOR_BG = (10, 15, 25)  # 深空赛博蓝
    COLOR_GRID = (25, 35, 55)  # 网格线
    COLOR_UAV = (0, 255, 200)  # 无人机青色
    COLOR_UAV_DEAD = (80, 80, 80)
    COLOR_BOSS = (255, 50, 80)  # 核心猩红
    COLOR_POISON = (138, 43, 226)  # 毒雾紫
    COLOR_WAVE = (255, 120, 0)  # 剑气橙
    COLOR_MISSILE = (255, 215, 0)  # 普攻金
    COLOR_OVERLOAD = (0, 255, 255)  # 过载青
    COLOR_HUD = (200, 220, 255)

    @staticmethod
    def draw_rotated_triangle(surface, color, center, radius, angle):
        """绘制带朝向的无人机(三角形)"""
        p1 = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
        p2 = (center[0] + radius * 0.8 * math.cos(angle + 2.5), center[1] + radius * 0.8 * math.sin(angle + 2.5))
        p3 = (center[0] + radius * 0.8 * math.cos(angle - 2.5), center[1] + radius * 0.8 * math.sin(angle - 2.5))
        pygame.draw.polygon(surface, color, [p1, p2, p3])
        pygame.draw.polygon(surface, (255, 255, 255), [p1, p2, p3], 1)  # 高亮边缘


# ==========================================
# 3. 核心评估与录制主循环
# ==========================================
def record_game_gifs(model_path, num_episodes=3, max_render_steps=600):
    print(f"[*] 正在加载神级微操模型: {model_path} ...")

    env = MicroBattleEnv()
    obs_dim = env.observation_space["agent_0"].shape[0]
    action_dim = env.action_space["agent_0"].shape[0]

    device = torch.device("cpu")
    actor = Actor(obs_dim, action_dim).to(device)

    try:
        actor.load_state_dict(torch.load(model_path, map_location=device))
        print("[*] 神经网络突触连接完毕！准备进入实机演示...")
    except Exception as e:
        print(f"[!] 模型加载失败，请检查路径。错误: {e}")
        return

    actor.eval()
    pygame.init()

    # 录制分辨率 (稍作缩小以控制 GIF 体积)
    scale = 0.7
    screen_size = int(Config.MAP_SIZE * scale)
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("UAV Swarm vs Goliath - RL Evaluation")

    font_sm = pygame.font.SysFont("consolas", 12, bold=True)
    font_md = pygame.font.SysFont("consolas", 16, bold=True)
    font_lg = pygame.font.SysFont("consolas", 48, bold=True)

    clock = pygame.time.Clock()

    # 用于保存生成的 GIF 路径
    output_dir = "game_records"
    os.makedirs(output_dir, exist_ok=True)

    for ep in range(1, num_episodes + 1):
        print(f"\n[>>>] 正在录制第 {ep}/{num_episodes} 局游戏...")
        obs_dict, _ = env.reset()

        frames = []
        running = True
        step_counter = 0

        # 视觉特效状态器
        time_tick = 0.0
        trails = {i: deque(maxlen=8) for i in range(Config.NUM_AGENTS)}  # 尾迹长度
        uav_angles = {i: 0.0 for i in range(Config.NUM_AGENTS)}

        while running and step_counter < max_render_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # --- RL 大脑决策 ---
            obs_tensor = torch.tensor(np.array(list(obs_dict.values())), dtype=torch.float32).to(device)
            with torch.no_grad():
                actions = actor(obs_tensor)

            action_np = actions.cpu().numpy()
            action_dict = {f"agent_{i}": action_np[i] for i in range(Config.NUM_AGENTS)}

            # 记录无人机偏航角 (根据输出的速度向量)
            for i in range(Config.NUM_AGENTS):
                if np.linalg.norm(action_np[i]) > 0.1:
                    uav_angles[i] = math.atan2(action_np[i][1], action_np[i][0])

            next_obs_dict, rewards, term, trunc, infos = env.step(action_dict)
            obs_dict = next_obs_dict
            step_counter += 1
            time_tick += 0.1

            # ==========================================
            # 画面渲染 (The Engine)
            # ==========================================
            screen.fill(VisualFX.COLOR_BG)

            # 1. 动态滚动网格
            grid_offset = (time_tick * 20) % 80
            for x in range(0, screen_size + 80, 80):
                pygame.draw.line(screen, VisualFX.COLOR_GRID, (x - grid_offset, 0), (x - grid_offset, screen_size))
            for y in range(0, screen_size + 80, 80):
                pygame.draw.line(screen, VisualFX.COLOR_GRID, (0, y - grid_offset), (screen_size, y - grid_offset))

            # 2. 威胁区 ( Danger Zone & Poison )
            boss_pos = (int(env.boss[0] * scale), int(env.boss[1] * scale))

            # 毒雾呼吸光环
            poison_r = int(Config.POISON_RADIUS * scale)
            breath_alpha = int(40 + 20 * math.sin(time_tick * 5))
            s_poison = pygame.Surface((screen_size, screen_size), pygame.SRCALPHA)
            pygame.draw.circle(s_poison, (*VisualFX.COLOR_POISON, breath_alpha), boss_pos, poison_r)
            screen.blit(s_poison, (0, 0))
            pygame.draw.circle(screen, (*VisualFX.COLOR_POISON, 150), boss_pos, poison_r, 1)

            # 剑气 / AOE
            if env.danger[0] == 1.0:
                danger_center = (int(env.danger[2] * scale), int(env.danger[3] * scale))
                if env.danger[1] == 1.0:
                    r = int(Config.AOE_RADIUS * scale)
                    pygame.draw.circle(screen, (255, 0, 0), danger_center, r, 2)

                elif env.danger[1] == 2.0:
                    current_radius = env.danger[6] * scale
                    thickness = Config.WAVE_THICKNESS * scale
                    dir_x, dir_y = env.danger[4], env.danger[5]
                    spread_cos = env.danger[7]

                    base_angle = math.atan2(dir_y, dir_x)
                    spread = math.acos(spread_cos)

                    points = []
                    for i in range(-10, 11):
                        a = base_angle + spread * (i / 10.0)
                        px = danger_center[0] + (current_radius + thickness) * math.cos(a)
                        py = danger_center[1] + (current_radius + thickness) * math.sin(a)
                        points.append((int(px), int(py)))
                    for i in range(10, -11, -1):
                        a = base_angle + spread * (i / 10.0)
                        px = danger_center[0] + (current_radius - thickness) * math.cos(a)
                        py = danger_center[1] + (current_radius - thickness) * math.sin(a)
                        points.append((int(px), int(py)))

                    s = pygame.Surface((screen_size, screen_size), pygame.SRCALPHA)
                    pygame.draw.polygon(s, (*VisualFX.COLOR_WAVE, 180), points)
                    screen.blit(s, (0, 0))
                    pygame.draw.polygon(screen, (255, 200, 100), points, 2)

            # 3. 导弹弹道
            for m in env.missiles:
                dist_to_boss = m['timer'] * Config.MISSILE_SPEED * scale
                src = (m['source_pos'][0] * scale, m['source_pos'][1] * scale)
                v_dir = np.array([src[0] - boss_pos[0], src[1] - boss_pos[1]])
                norm = np.linalg.norm(v_dir)
                if norm > 0: v_dir /= norm

                m_pos = (int(boss_pos[0] + v_dir[0] * dist_to_boss), int(boss_pos[1] + v_dir[1] * dist_to_boss))
                color = VisualFX.COLOR_OVERLOAD if m['is_overload'] else VisualFX.COLOR_MISSILE
                radius = 5 if m['is_overload'] else 2

                # 绘制发光弹道
                s_glow = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
                pygame.draw.circle(s_glow, (*color, 100), (radius * 2, radius * 2), radius * 2)
                screen.blit(s_glow, (m_pos[0] - radius * 2, m_pos[1] - radius * 2))
                pygame.draw.circle(screen, (255, 255, 255), m_pos, radius)

            # 4. Boss 渲染
            boss_r = int(Config.BOSS_RADIUS * scale)
            hp_ratio = max(0, env.boss[2]) / Config.BOSS_MAX_HP

            # 护盾旋转环
            shield_prob = Config.BOSS_IMMUNE_BASE + (1 - hp_ratio) * (Config.BOSS_IMMUNE_MAX - Config.BOSS_IMMUNE_BASE)
            ring_angle = time_tick * 3
            px = boss_pos[0] + (boss_r + 10) * math.cos(ring_angle)
            py = boss_pos[1] + (boss_r + 10) * math.sin(ring_angle)
            pygame.draw.circle(screen, (0, 200, 255), (int(px), int(py)), 5)
            pygame.draw.circle(screen, (0, 100, 255), boss_pos, boss_r + 10, max(1, int(shield_prob * 10)))

            # 本体与朝向
            boss_color = (255, 180, 0) if env.boss[3] == 1.0 else VisualFX.COLOR_BOSS
            pygame.draw.circle(screen, boss_color, boss_pos, boss_r)
            pygame.draw.circle(screen, (30, 0, 0), boss_pos, boss_r, 4)

            dir_x, dir_y = env.boss[6], env.boss[7]
            eye_pos = (int(boss_pos[0] + dir_x * boss_r * 0.8), int(boss_pos[1] + dir_y * boss_r * 0.8))
            pygame.draw.circle(screen, (255, 255, 255), eye_pos, 6)

            # UI: Boss HP
            pygame.draw.rect(screen, (80, 0, 0), (boss_pos[0] - 50, boss_pos[1] - boss_r - 25, 100, 8))
            pygame.draw.rect(screen, (255, 50, 50), (boss_pos[0] - 50, boss_pos[1] - boss_r - 25, 100 * hp_ratio, 8))
            pygame.draw.rect(screen, (255, 255, 255), (boss_pos[0] - 50, boss_pos[1] - boss_r - 25, 100, 8), 1)

            # 5. 无人机渲染 (Agent)
            for i in range(Config.NUM_AGENTS):
                agent_pos = (int(env.agents[i, 0] * scale), int(env.agents[i, 1] * scale))
                is_dead = env.agents[i, 4] == 1
                color = VisualFX.COLOR_UAV_DEAD if is_dead else VisualFX.COLOR_UAV
                a_radius = int(Config.AGENT_RADIUS * scale)

                # 更新并绘制尾迹
                if not is_dead:
                    trails[i].append(agent_pos)
                    if len(trails[i]) > 1:
                        points = list(trails[i])
                        for j in range(len(points) - 1):
                            alpha = int(255 * (j / len(points)))
                            pygame.draw.line(screen, (*color, alpha), points[j], points[j + 1], 2)

                    # 绘制战机本体
                    VisualFX.draw_rotated_triangle(screen, color, agent_pos, a_radius + 4, uav_angles[i])

                    # 绘制被锁定预警连线
                    if env.boss[5] == i:
                        pygame.draw.line(screen, (255, 50, 50), agent_pos, boss_pos, 1)
                        pygame.draw.circle(screen, (255, 0, 0), agent_pos, a_radius + 8, 1)

                    # UI: 无人机状态 (血条 + 充能大招)
                    agent_hp_ratio = max(0, env.agents[i, 2]) / Config.AGENT_MAX_HP
                    pygame.draw.rect(screen, (100, 0, 0), (agent_pos[0] - 20, agent_pos[1] - a_radius - 15, 40, 4))
                    pygame.draw.rect(screen, (0, 255, 100),
                                     (agent_pos[0] - 20, agent_pos[1] - a_radius - 15, 40 * agent_hp_ratio, 4))

                    charge_ratio = env.agents[i, 5]
                    c_color = VisualFX.COLOR_OVERLOAD if charge_ratio >= 1.0 else (255, 165, 0)
                    pygame.draw.rect(screen, (50, 50, 50), (agent_pos[0] - 20, agent_pos[1] - a_radius - 8, 40, 3))
                    pygame.draw.rect(screen, c_color,
                                     (agent_pos[0] - 20, agent_pos[1] - a_radius - 8, 40 * charge_ratio, 3))

            # 6. HUD 抬头显示器
            hud_texts = [
                f">> UAV SWARM UPLINK",
                f"STATUS: {'ENGAGED' if not any(term.values()) else 'TERMINATED'}",
                f"BOSS INTEGRITY: {env.boss[2]:.0f}/{Config.BOSS_MAX_HP:.0f}",
                f"TIME ELAPSED: {time_tick:.1f} s"
            ]
            for idx, text in enumerate(hud_texts):
                rendered = font_md.render(text, True, VisualFX.COLOR_HUD)
                screen.blit(rendered, (20, 20 + idx * 25))

            pygame.display.flip()

            # --- 抓取帧存为 GIF (关键帧抽样压缩) ---
            # 每 2 帧抓取一次，相当于 15 FPS 的 GIF，极大地减小了文件体积，同时保持了微操的观赏性
            if step_counter % 2 == 0:
                frame = pygame.surfarray.array3d(screen)
                frame = np.transpose(frame, (1, 0, 2))  # Pygame 表面矩阵转置为标准图像矩阵
                frames.append(frame)

            clock.tick(30)  # 游戏以 30 FPS 运行，保证录制过程的物理流畅度

            # 终局判定
            if any(term.values()) or any(trunc.values()):
                res_str = "TARGET DESTROYED" if env.boss[2] <= 0 else "SWARM ANNIHILATED"
                res_color = (0, 255, 200) if env.boss[2] <= 0 else (255, 50, 50)
                res_text = font_lg.render(res_str, True, res_color)
                screen.blit(res_text, (screen_size // 2 - res_text.get_width() // 2,
                                       screen_size // 2 - res_text.get_height() // 2))
                pygame.display.flip()

                # 额外停留 1 秒抓取结局画面
                for _ in range(15):
                    frame = pygame.surfarray.array3d(screen)
                    frames.append(np.transpose(frame, (1, 0, 2)))
                running = False

        # === 保存当前对局为 GIF ===
        gif_filename = os.path.join(output_dir, f"battle_record_ep{ep}.gif")
        print(f"[*] 对局结束，正在压缩并导出 GIF (共 {len(frames)} 帧)...")
        # fps=15 与我们两帧一抽样的逻辑匹配，保证播放速度与真实时间一致
        imageio.mimsave(gif_filename, frames, fps=15, format='GIF', loop=0)
        print(f"[+] 录制成功！文件已保存至: {gif_filename}")

    pygame.quit()
    print("\n[✔] 所有实机测试录像已完成，可以在 game_records 文件夹中查看。")


if __name__ == "__main__":
    # >>> 替换为你训练好的模型路径 <<<
    MODEL_PATH = "checkpoints/mappo_actor_step_1001472.pth"

    if os.path.exists(MODEL_PATH):
        # 录制 3 局，每局最长渲染 600 步（60秒真实世界时间）
        record_game_gifs(MODEL_PATH, num_episodes=3, max_render_steps=600)
    else:
        print(f"[!] 找不到权重文件 {MODEL_PATH}")
        print("请在 train.py 跑出结果后，将对应 .pth 文件的路径填入上方代码！")