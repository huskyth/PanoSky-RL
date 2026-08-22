import math
import numpy as np
import pandas as pd
import pygame
import sys
from env import MicroBattleEnv, Config


def test_spaces_and_step(env):
    """1. 测试新维度的状态空间、动作空间与 step() 交互"""
    print("=" * 60)
    print("开始测试：升级版环境状态空间、动作空间与 step() 交互")
    print("=" * 60)

    obs, info = env.reset()
    agent_id = "agent_0"

    print(f"[{agent_id}] 观测空间维度: {env.observation_space[agent_id].shape} (预期为 29 维)")
    print(f"[{agent_id}] 动作空间维度: {env.action_space[agent_id].shape}")
    print(f"[{agent_id}] 初始观测值示例 (前5维 - 包含充能): {obs[agent_id][:5]}")

    actions = {f"agent_{i}": env.action_space[f"agent_{i}"].sample() for i in range(Config.NUM_AGENTS)}
    next_obs, rewards, terminations, truncations, infos = env.step(actions)

    print(f"[{agent_id}] 随机步后奖励: {rewards[agent_id]:.4f}")
    print(f"[{agent_id}] Info 字典内容: {infos[agent_id]}")
    print(">> 基础交互与维度升级测试通过！\n")


def test_rendering(env):
    """2. 使用 Pygame 渲染高级游戏机制 (持续运行 1000 步)"""
    print("=" * 60)
    print("开始测试：Pygame 可视化 (Boss朝向、扇形扫射、导弹、充能条)")
    print("=" * 60)

    pygame.init()
    scale = 0.8
    screen_size = int(Config.MAP_SIZE * scale)
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Micro Battle V2 - Advanced Mechanics Test")
    clock = pygame.time.Clock()

    env.reset()
    running = True
    steps = 0
    max_render_steps = 1000

    while running and steps < max_render_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        actions = {f"agent_{i}": env.action_space[f"agent_{i}"].sample() for i in range(Config.NUM_AGENTS)}
        _, _, term, trunc, _ = env.step(actions)
        steps += 1

        screen.fill((240, 240, 240))

        # ==========================================
        # 绘制动态威胁区 (定点 AOE & 动态冲击波)
        # ==========================================
        if env.danger[0] == 1.0:
            danger_center = (int(env.danger[2] * scale), int(env.danger[3] * scale))

            if env.danger[1] == 1.0:  # 圆圈 AOE
                danger_radius = int(Config.AOE_RADIUS * scale)
                s = pygame.Surface((screen_size, screen_size), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 0, 0, 80), danger_center, danger_radius)
                screen.blit(s, (0, 0))
                pygame.draw.circle(screen, (255, 0, 0), danger_center, danger_radius, 2)

            elif env.danger[1] == 2.0:  # 动态移动的随机宽度剑气
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
                pygame.draw.polygon(s, (255, 69, 0, 150), points)
                screen.blit(s, (0, 0))
                pygame.draw.polygon(screen, (255, 0, 0), points, 2)

        # ==========================================
        # 绘制 Boss 与朝向装甲区
        # ==========================================
        boss_pos = (int(env.boss[0] * scale), int(env.boss[1] * scale))
        boss_radius = int(Config.BOSS_RADIUS * scale)

        # 护盾指示 (根据血量变蓝)
        hp_ratio = max(0, env.boss[2]) / Config.BOSS_MAX_HP
        shield_alpha = int(
            (Config.BOSS_IMMUNE_BASE + (1 - hp_ratio) * (Config.BOSS_IMMUNE_MAX - Config.BOSS_IMMUNE_BASE)) * 255)
        pygame.draw.circle(screen, (0, 100, 255, shield_alpha), boss_pos, boss_radius + 5, 2)

        # Boss 本体
        pygame.draw.circle(screen, (150, 30, 30), boss_pos, boss_radius)
        if env.boss[3] == 1.0:  # 前摇变黄
            pygame.draw.circle(screen, (255, 200, 0), boss_pos, boss_radius + 2, 3)

        # 朝向指示线 (黑色)
        dir_x, dir_y = env.boss[6], env.boss[7]
        face_end = (int(boss_pos[0] + dir_x * boss_radius * 1.5), int(boss_pos[1] + dir_y * boss_radius * 1.5))
        pygame.draw.line(screen, (0, 0, 0), boss_pos, face_end, 4)

        # Boss 血条
        pygame.draw.rect(screen, (200, 0, 0), (boss_pos[0] - 30, boss_pos[1] - 55, 60, 6))
        pygame.draw.rect(screen, (0, 255, 0), (boss_pos[0] - 30, boss_pos[1] - 55, 60 * hp_ratio, 6))

        # ==========================================
        # 绘制 Agents 与充能条
        # ==========================================
        for i in range(Config.NUM_AGENTS):
            if env.agents[i, 4] == 0:
                agent_pos = (int(env.agents[i, 0] * scale), int(env.agents[i, 1] * scale))
                pygame.draw.circle(screen, (30, 100, 200), agent_pos, int(Config.AGENT_RADIUS * scale))

                if env.boss[5] == i:  # 仇恨标记
                    pygame.draw.circle(screen, (255, 0, 0), agent_pos, int(Config.AGENT_RADIUS * scale) + 4, 2)

                # 血条
                agent_hp_ratio = max(0, env.agents[i, 2]) / Config.AGENT_MAX_HP
                pygame.draw.rect(screen, (200, 0, 0), (agent_pos[0] - 15, agent_pos[1] - 25, 30, 4))
                pygame.draw.rect(screen, (0, 255, 0), (agent_pos[0] - 15, agent_pos[1] - 25, 30 * agent_hp_ratio, 4))

                # 充能条 (黄色)
                charge_ratio = env.agents[i, 5]
                pygame.draw.rect(screen, (100, 100, 100), (agent_pos[0] - 15, agent_pos[1] - 20, 30, 3))
                if charge_ratio >= 1.0:
                    pygame.draw.rect(screen, (0, 255, 255), (agent_pos[0] - 15, agent_pos[1] - 20, 30, 3))  # 满充能青色
                else:
                    pygame.draw.rect(screen, (255, 215, 0),
                                     (agent_pos[0] - 15, agent_pos[1] - 20, 30 * charge_ratio, 3))

        # ==========================================
        # 绘制飞行中的导弹
        # ==========================================
        for m in env.missiles:
            # 估算导弹当前位置
            dist_to_boss = m['timer'] * Config.MISSILE_SPEED * scale
            src = (m['source_pos'][0] * scale, m['source_pos'][1] * scale)
            b_pos = (env.boss[0] * scale, env.boss[1] * scale)

            # 从 Boss 反推导弹位置 (因为导弹最终要落到 Boss 身上)
            v_dir = np.array([src[0] - b_pos[0], src[1] - b_pos[1]])
            norm = np.linalg.norm(v_dir)
            if norm > 0: v_dir /= norm

            m_pos = (int(b_pos[0] + v_dir[0] * dist_to_boss), int(b_pos[1] + v_dir[1] * dist_to_boss))

            color = (0, 255, 255) if m['is_overload'] else (255, 140, 0)
            radius = 6 if m['is_overload'] else 3
            pygame.draw.circle(screen, color, m_pos, radius)

        pygame.display.flip()
        clock.tick(15)  # 保持 15 帧，让你能看清扇形扫描和导弹飞行

        if any(term.values()) or any(trunc.values()):
            env.reset()

    pygame.quit()
    print(">> 渲染测试结束！\n")


def test_statistics(env):
    """3. 极限压测与 Pandas 数据分析 (适配延迟伤害)"""
    print("=" * 60)
    print("开始测试：10000 步随机策略压测与奖励分布分析")
    print("=" * 60)

    env.reset()
    reward_data = []

    for step in range(100000):
        # 每隔 200 步，把小兵分散传送到 Boss 周围 100 距离的圆圈上
        # 这样既能测试正面减伤，也能测试背刺全伤，还能触发充能和射击
        if step % 200 == 0:
            for i in range(Config.NUM_AGENTS):
                if env.agents[i, 4] == 0:
                    angle = i * (2 * math.pi / Config.NUM_AGENTS)
                    env.agents[i, 0] = env.boss[0] + 100 * math.cos(angle)
                    env.agents[i, 1] = env.boss[1] + 100 * math.sin(angle)
                    env.agents[i, 2] = Config.AGENT_MAX_HP
                    env.agents[i, 5] = 1.0  # 直接给满充能测试大招
            env.boss[2] = Config.BOSS_MAX_HP

        actions = {f"agent_{i}": env.action_space[f"agent_{i}"].sample() for i in range(Config.NUM_AGENTS)}
        _, _, term, trunc, infos = env.step(actions)

        for i in range(Config.NUM_AGENTS):
            if env.agents[i, 4] == 0:
                info = infos[f"agent_{i}"]
                reward_data.append({
                    "r_step": info["r_step"],
                    "r_nav": info["r_nav"],
                    "r_aoe": info["r_aoe"],
                    "r_surround": info["r_surround"],
                    "r_dmg": info["r_dmg"],
                    "r_hurt": info.get("r_hurt", 0)  # 加上受伤惩罚
                })

        if any(term.values()) or any(trunc.values()):
            env.reset()

    df = pd.DataFrame(reward_data)
    stats = df.describe().T
    stats['var'] = df.var()

    columns_order = ['mean', 'var', 'min', '25%', '50%', '75%', 'max']
    final_stats = stats[columns_order]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print("各奖励组件的统计分布 (基于强制接战与大招的随机策略)：")
    print(final_stats)
    print("\n[分析提示]：")
    print("1. 观察 r_dmg 的最大值。由于满充能大招伤害 x3，背刺判定生效，最大单步伤害奖励可能会有所上升。")
    print("2. 扇形 AOE 的惩罚比之前的圆圈更大且判定范围更长，请确保 r_aoe 的惩罚下限没有超过 [-2.0]。")


if __name__ == "__main__":
    test_env = MicroBattleEnv()

    test_spaces_and_step(test_env)
    test_rendering(test_env)
    test_statistics(test_env)