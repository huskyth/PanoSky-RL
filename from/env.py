import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class Config:
    """集中保存所有环境、游戏与奖励参数"""

    # ==========================================
    # 空间与时间约束
    # ==========================================
    MAP_SIZE = 1000.0  # 连续二维正方形地图的边长。四周有空气墙，限制了极限拉扯的空间。
    MAX_STEPS = 2000  # 回合最大步数（Truncation 截断）。防止智能体学会“永远逃跑不接战”的消极次优解。

    # ==========================================
    # 智能体 (RL Agent) 参数
    # ==========================================
    NUM_AGENTS = 3  # 我方敏捷单位数量。
    AGENT_RADIUS = 15.0  # 智能体物理碰撞半径。
    AGENT_MAX_SPEED = 60.0  # 智能体最大移动速度（像素/秒），提供了对抗冲击波和拉扯 Boss 的机动性基础。
    AGENT_MAX_HP = 100.0  # 智能体最大血量。容错率极低，吃满两三个技能就会阵亡。
    AGENT_ATK_RANGE = 400.0  # 普攻射程。足够大的射程让智能体能在安全距离外“风筝” Boss。
    AGENT_ATK_DMG = 30.0  # 单次普攻基础伤害。
    AGENT_CD_MAX = 1.5  # 普攻冷却时间。攻击后必须等待，这构成了 Hit & Run（走砍）微操的物理底层动机。

    # ==========================================
    # 智能体进阶战斗机制 (延迟与大招)
    # ==========================================
    MISSILE_SPEED = 500.0  # 导弹飞行速度。数值越高，伤害延迟越短，神经网络越容易将“开火动作”与“伤害奖励”建立因果联系。
    CHARGE_MAX_STEPS = 100  # 大招充能所需存活步数（相当于存活 10 秒）。极大地鼓励了“保命优先”的战术。
    OVERLOAD_DMG_MULT = 3.0  # 满充能过载打击的伤害倍率。一发顶三发，配合背刺效果毁天灭地。

    # ==========================================
    # Boss 参数 (内置 AI)
    # ==========================================
    BOSS_RADIUS = 40.0  # Boss 物理碰撞半径。大体积使其容易被卡位或绕背。
    BOSS_MAX_SPEED = 25.0  # Boss 移速（远低于智能体），使其在被风筝时处于绝对劣势，只能靠技能弥补。
    BOSS_MAX_HP = 1500.0  # Boss 最大血量。高血量意味着需要长线的协同输出，防止偶然的一次集火直接结束游戏。
    BOSS_PRE_CAST_TIME = 1.5  # 技能施法前摇（秒）。给智能体提供极其宝贵的反应和撤退窗口期。
    BOSS_CD_MAX = 2.5  # Boss 技能冷却时间（秒）。

    # ==========================================
    # Boss 进阶战斗机制 (毒雾与护盾)
    # ==========================================
    POISON_RADIUS = 100.0  # 毒雾光环半径。绝对的“物理禁区”。
    POISON_DMG_PER_SEC = 20.0  # 毒雾每秒伤害。通过持续的高额掉血惩罚，彻底粉碎智能体“贴脸速刷伤害”的企图。
    BOSS_IMMUNE_BASE = 0.10  # Boss 满血时的基础弹道免疫/格挡概率 (10%)。
    BOSS_IMMUNE_MAX = 0.30  # Boss 濒死时的极限免疫概率 (30%)。引入环境随机性，打破固定输出轴的“背板”行为。
    FRONTAL_COS_THRESHOLD = 0.707  # 正面装甲判定阈值（约45度）。在此角度内攻击伤害减半，强烈引导“正面拉扯、侧背输出”的协同战术。

    # ==========================================
    # 动态威胁区 (Danger Zone: AOE & 冲击波)
    # ==========================================
    AOE_RADIUS = 60.0  # 锁定目标的定点红色爆炸圈半径。
    WAVE_SPEED = 60.0  # 动态冲击波扩散速度。恰好等于智能体最大移速，意味着往后退是等死，必须横向走位（切向逃逸）。
    WAVE_MAX_RADIUS = 500.0  # 冲击波最远波及范围。出了这个范围剑气自动消散。
    WAVE_THICKNESS = 45.0  # 冲击波的物理判定厚度。加厚判定意味着智能体需要更长的时间/距离来穿透剑气，增加了规避难度。
    DANGER_DELAY = 1.5  # AOE 红圈从出现到爆炸的延迟（秒）。
    DANGER_DMG = 20.0  # 任何危险区命中的基础伤害。

    DT = 0.1  # 物理引擎的单步时间跨度（0.1秒 = 10Hz）。数值越小，物理模拟越精细，但训练相同的真实时间需要更多步数。

    # ==========================================
    # RL 奖励塑形 (Reward Shaping) 系数
    # ==========================================
    # 【惩罚项】
    R_STEP = -0.02  # 生存时间惩罚。驱使智能体保持进取心，防止消极避战。
    R_HURT_COEF = 0.02  # 受伤惩罚系数。例如受到 20 点伤害会扣除 0.4 分，从价值计算上杜绝了“以血换输出”的策略。
    R_DEATH = -20.0  # 阵亡的终极惩罚。将其强制踢出对局。

    # 【密集引导项 (Dense Shaping)】
    R_NAV_COEF = 0.02  # 靠近 Boss 的引导奖励。只在射程外生效，防止挂机。
    R_AOE_DANGER_COEF = 0.02  # 危险区压迫惩罚。距离 AOE 中心或冲击波面越近惩罚越大，提供“横向逃离”的平滑梯度。
    R_SURROUND_COEF = 0.015  # 完美包围阵型奖励。注意：0.015 < abs(-0.02)，保证了“只围不打”的净收益为负，堵死了和平主义刷分漏洞。

    # 【稀疏事件项 (Sparse Events)】
    R_DMG_BASE = 0.01  # 单点伤害奖励的基础乘区。将单步极值压制在 [-2, 2] 之间，防止 Critic 网络价值爆炸。
    R_DMG_SCALE = 1.0  # 斩杀附加系数。Boss 血量越低，伤害奖励越高（最高 2 倍），鼓励在 Boss 狂暴期集火速杀。
    R_AOE_MISS = 5.0  # 骗技能奖励。如果全队无人被这次 Boss 技能打中，全队共享大奖，极度鼓励“拉扯与走位”。
    R_KILL = 100.0  # 击杀 Boss 游戏胜利的终极奖励。


class Plot:
    @staticmethod
    def plot_learning_curve(data, title="Learning Curve", ylabel="Reward", window=100):
        plt.figure(figsize=(10, 5))
        plt.plot(data, alpha=0.3, color='blue', label='Raw')
        if len(data) >= window:
            smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
            plt.plot(np.arange(window - 1, len(data)), smoothed, color='red', label=f'MA ({window})')
        plt.title(title)
        plt.xlabel("Episodes")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()


class MicroBattleEnv(gym.Env):
    def __init__(self, config=Config):
        super().__init__()
        self.cfg = config

        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for i in range(self.cfg.NUM_AGENTS)
        })

        obs_dim = 30
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for i in range(self.cfg.NUM_AGENTS)
        })

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self.step_count = 0

        self.agents = np.zeros((self.cfg.NUM_AGENTS, 6), dtype=np.float32)
        for i in range(self.cfg.NUM_AGENTS):
            self.agents[i, 0] = np.random.uniform(50, 200) if np.random.rand() > 0.5 else np.random.uniform(800, 950)
            self.agents[i, 1] = np.random.uniform(50, 200) if np.random.rand() > 0.5 else np.random.uniform(800, 950)
            self.agents[i, 2] = self.cfg.AGENT_MAX_HP
            self.agents[i, 3] = 0.0
            self.agents[i, 4] = 0.0
            self.agents[i, 5] = 0.0

        self.boss = np.array([
            self.cfg.MAP_SIZE / 2, self.cfg.MAP_SIZE / 2,
            self.cfg.BOSS_MAX_HP, 0.0, 0.0, -1.0, 0.0, 1.0
        ], dtype=np.float32)

        self.danger = np.array([0.0] * 11, dtype=np.float32)
        self.missiles = []
        self.prev_dists = self._get_boss_dists()
        return self._get_obs(), {}

    def step(self, action_dict):
        self.step_count += 1
        rewards = {f"agent_{i}": 0.0 for i in range(self.cfg.NUM_AGENTS)}
        infos = {
            f"agent_{i}": {"r_step": 0, "r_nav": 0, "r_aoe": 0, "r_surround": 0, "r_dmg": 0, "r_hurt": 0, "r_slack": 0}
            for i in range(self.cfg.NUM_AGENTS)}

        dists_to_boss = self._get_boss_dists()
        hp_before = self.agents[:, 2].copy()

        # 1-5 步的物理、毒雾、导弹和 Boss 技能判定保持不变
        self._apply_physics(action_dict)
        self._apply_poison_aura()
        dmg_dealt = self._process_missiles()
        aoe_missed = self._update_boss_and_danger(dists_to_boss)
        self._agent_auto_attacks_and_charge(dists_to_boss)

        new_dists = self._get_boss_dists()

        # === 修复 1：严格的包围网判定 ===
        r_dmg_shared = 0.0
        if dmg_dealt > 0:
            hp_ratio = self.boss[2] / self.cfg.BOSS_MAX_HP
            kill_coef = 1.0 + self.cfg.R_DMG_SCALE * (1.0 - hp_ratio)
            r_dmg_shared = dmg_dealt * self.cfg.R_DMG_BASE * kill_coef

        r_aoe_miss_shared = self.cfg.R_AOE_MISS if aoe_missed else 0.0

        r_surround_shared = 0.0
        alive_idx = np.where(self.agents[:, 4] == 0)[0]
        # 只计算距离小于 500.0 (合法作战半径) 的存活者构成的包围网
        active_idx = [i for i in alive_idx if new_dists[i] < 500.0]
        if len(active_idx) > 1:
            vecs = []
            for i in active_idx:
                v = self.agents[i, :2] - self.boss[:2]
                norm = np.linalg.norm(v)
                if norm > 0: vecs.append(v / norm)
            if vecs:
                vec_sum = np.sum(vecs, axis=0)
                r_surround_shared = self.cfg.R_SURROUND_COEF * (1.0 - np.linalg.norm(vec_sum) / len(vecs))

        # === 修复 2：个人绩效与摸鱼惩罚 ===
        for i in range(self.cfg.NUM_AGENTS):
            if self.agents[i, 4] == 1: continue

            r_step = self.cfg.R_STEP

            # [关键修复：专属向心力] 只要你*自己*不在射程内，就必须获得向内靠拢的梯度
            r_nav = 0.0
            if new_dists[i] > self.cfg.AGENT_ATK_RANGE:
                r_nav = self.cfg.R_NAV_COEF * (self.prev_dists[i] - new_dists[i])

            # [关键修复：摸鱼惩罚 (Anti-Slacking)]
            r_slack = 0.0
            combat_radius = 500.0  # 允许拉扯的安全距离
            # 如果你游离在战场边缘 (距离>500)，并且 Boss 当前并没有锁定追杀你
            if new_dists[i] > combat_radius and self.boss[5] != i:
                # 每远离 100 像素，每步施加 -0.05 的重度惩罚！
                r_slack = -0.05 * ((new_dists[i] - combat_radius) / 100.0)

            r_danger_penalty = 0.0
            if self.danger[0] == 1.0:
                vec_to_agent = self.agents[i, :2] - self.danger[2:4]
                dist_to_danger = np.linalg.norm(vec_to_agent)

                if self.danger[1] == 1.0:
                    if dist_to_danger < self.cfg.AOE_RADIUS:
                        r_danger_penalty = -self.cfg.R_AOE_DANGER_COEF * (1.0 - dist_to_danger / self.cfg.AOE_RADIUS)

                elif self.danger[1] == 2.0:
                    current_radius = self.danger[6]
                    spread_cos = self.danger[7]
                    if dist_to_danger > 0: vec_to_agent /= dist_to_danger
                    cos_theta = np.dot(vec_to_agent, self.danger[4:6])

                    if cos_theta >= spread_cos and dist_to_danger >= current_radius - self.cfg.WAVE_THICKNESS:
                        dist_to_wave = dist_to_danger - current_radius
                        if dist_to_wave < 350.0:
                            r_danger_penalty = -self.cfg.R_AOE_DANGER_COEF * (1.0 - max(0, dist_to_wave) / 350.0)

            # 加入摸鱼惩罚
            dense_r = r_step + r_nav + r_danger_penalty + r_surround_shared + r_dmg_shared + r_slack
            dense_r = np.clip(dense_r, -2.0, 2.0)
            rewards[f"agent_{i}"] = dense_r + r_aoe_miss_shared

            infos[f"agent_{i}"]["r_step"] = r_step
            infos[f"agent_{i}"]["r_nav"] = r_nav
            infos[f"agent_{i}"]["r_aoe"] = r_danger_penalty + r_aoe_miss_shared
            infos[f"agent_{i}"]["r_surround"] = r_surround_shared
            infos[f"agent_{i}"]["r_dmg"] = r_dmg_shared
            infos[f"agent_{i}"]["r_slack"] = r_slack

            # 受伤惩罚
            hp_lost = hp_before[i] - self.agents[i, 2]
            if hp_lost > 0:
                r_hurt = -hp_lost * self.cfg.R_HURT_COEF
                rewards[f"agent_{i}"] += r_hurt
                infos[f"agent_{i}"]["r_hurt"] = r_hurt

        self.prev_dists = new_dists
        terminations = {f"agent_{i}": False for i in range(self.cfg.NUM_AGENTS)}
        truncations = {f"agent_{i}": False for i in range(self.cfg.NUM_AGENTS)}

        self.boss[2] = max(0.0, self.boss[2])
        boss_dead = self.boss[2] <= 0
        all_dead = np.all(self.agents[:, 4] == 1)
        is_truncated = self.step_count >= self.cfg.MAX_STEPS

        if boss_dead:
            for i in range(self.cfg.NUM_AGENTS):
                if self.agents[i, 4] == 0: rewards[f"agent_{i}"] += self.cfg.R_KILL
                terminations[f"agent_{i}"] = True
        elif all_dead:
            for i in range(self.cfg.NUM_AGENTS): terminations[f"agent_{i}"] = True
        elif is_truncated:
            for i in range(self.cfg.NUM_AGENTS): truncations[f"agent_{i}"] = True

        infos["env_state"] = {"boss_hp": self.boss[2], "step": self.step_count}
        return self._get_obs(), rewards, terminations, truncations, infos

    def _apply_physics(self, action_dict):
        # 1. 应用速度
        for i in range(self.cfg.NUM_AGENTS):
            if self.agents[i, 4] == 1: continue
            self.agents[i, 3] = max(0.0, self.agents[i, 3] - self.cfg.DT)
            act = action_dict[f"agent_{i}"]
            velocity = act * self.cfg.AGENT_MAX_SPEED
            self.agents[i, :2] += velocity * self.cfg.DT
            self.agents[i, :2] = np.clip(self.agents[i, :2], self.cfg.AGENT_RADIUS,
                                         self.cfg.MAP_SIZE - self.cfg.AGENT_RADIUS)

        # 2. 智能体之间的简单排斥
        for i in range(self.cfg.NUM_AGENTS):
            if self.agents[i, 4] == 1: continue
            for j in range(i + 1, self.cfg.NUM_AGENTS):
                if self.agents[j, 4] == 1: continue
                diff = self.agents[i, :2] - self.agents[j, :2]
                dist = np.linalg.norm(diff)
                if dist < self.cfg.AGENT_RADIUS * 2:
                    overlap = self.cfg.AGENT_RADIUS * 2 - dist
                    push = (diff / (dist + 1e-8)) * (overlap / 2.0)
                    self.agents[i, :2] += push
                    self.agents[j, :2] -= push

        # 3. [修复核心] 绝对刚体隔离：防穿模
        min_boss_dist = self.cfg.BOSS_RADIUS + self.cfg.AGENT_RADIUS
        for i in range(self.cfg.NUM_AGENTS):
            if self.agents[i, 4] == 1: continue
            vec_to_boss = self.agents[i, :2] - self.boss[:2]
            dist_to_boss = np.linalg.norm(vec_to_boss)
            # 如果嵌入了 Boss 体内，强制重置坐标到边缘
            if dist_to_boss < min_boss_dist:
                if dist_to_boss == 0:
                    vec_to_boss = np.array([1.0, 0.0])  # 防止除零
                else:
                    vec_to_boss /= dist_to_boss
                self.agents[i, :2] = self.boss[:2] + vec_to_boss * (min_boss_dist + 0.1)

        # 4. 更新 Boss 朝向
        target_idx = int(self.boss[5])
        if target_idx >= 0 and self.agents[target_idx, 4] == 0:
            v_dir = self.agents[target_idx, :2] - self.boss[:2]
            norm = np.linalg.norm(v_dir)
            if norm > 0: self.boss[6:8] = v_dir / norm

    def _apply_poison_aura(self):
        """处理 Boss 周围的致命毒雾"""
        for i in range(self.cfg.NUM_AGENTS):
            if self.agents[i, 4] == 1: continue
            dist = np.linalg.norm(self.agents[i, :2] - self.boss[:2])
            if dist < self.cfg.POISON_RADIUS:
                # 处于毒雾中，受到高额秒伤
                dmg = self.cfg.POISON_DMG_PER_SEC * self.cfg.DT
                self.agents[i, 2] -= dmg
                if self.agents[i, 2] <= 0:
                    self.agents[i, 4] = 1.0

    def _process_missiles(self):
        total_dmg = 0.0
        remaining_missiles = []
        boss_hp_ratio = self.boss[2] / self.cfg.BOSS_MAX_HP
        immunity_prob = self.cfg.BOSS_IMMUNE_BASE + (1.0 - boss_hp_ratio) * (
                    self.cfg.BOSS_IMMUNE_MAX - self.cfg.BOSS_IMMUNE_BASE)

        for m in self.missiles:
            m['timer'] -= self.cfg.DT
            if m['timer'] <= 0:
                if self.boss[2] > 0:
                    is_immune = (np.random.rand() < immunity_prob) and not m['is_overload']
                    if not is_immune:
                        base_dmg = self.cfg.AGENT_ATK_DMG
                        if m['is_overload']: base_dmg *= self.cfg.OVERLOAD_DMG_MULT

                        v_atk = m['source_pos'] - self.boss[:2]
                        norm = np.linalg.norm(v_atk)
                        if norm > 0: v_atk /= norm
                        cos_theta = np.dot(v_atk, self.boss[6:8])

                        if cos_theta > self.cfg.FRONTAL_COS_THRESHOLD: base_dmg *= 0.5
                        total_dmg += base_dmg
            else:
                remaining_missiles.append(m)

        self.missiles = remaining_missiles
        self.boss[2] -= total_dmg
        return total_dmg

    def _agent_auto_attacks_and_charge(self, dists_to_boss):
        for i in range(self.cfg.NUM_AGENTS):
            if self.agents[i, 4] == 1: continue
            self.agents[i, 5] = min(1.0, self.agents[i, 5] + 1.0 / self.cfg.CHARGE_MAX_STEPS)

            if dists_to_boss[i] <= self.cfg.AGENT_ATK_RANGE and self.agents[i, 3] == 0.0:
                flight_time = dists_to_boss[i] / self.cfg.MISSILE_SPEED
                is_overload = self.agents[i, 5] >= 1.0

                self.missiles.append({
                    'timer': flight_time,
                    'source_pos': self.agents[i, :2].copy(),
                    'agent_id': i,
                    'is_overload': is_overload
                })
                self.agents[i, 3] = self.cfg.AGENT_CD_MAX
                if is_overload: self.agents[i, 5] = 0.0

    def _update_boss_and_danger(self, dists_to_boss):
        aoe_missed = False
        boss_state = self.boss[3]

        if boss_state == 0.0:
            alive_idx = np.where(self.agents[:, 4] == 0)[0]
            if len(alive_idx) > 0:
                closest_idx = alive_idx[np.argmin(dists_to_boss[alive_idx])]
                current_target = int(self.boss[5])

                if current_target in alive_idx and dists_to_boss[closest_idx] >= dists_to_boss[current_target] * 0.75:
                    target_idx = current_target
                else:
                    target_idx = closest_idx
                    self.boss[5] = target_idx

                atk_type = 1.0 if np.random.rand() < 0.5 else 2.0
                if atk_type == 1.0:
                    self.danger = np.array([1.0, 1.0, self.agents[target_idx, 0], self.agents[target_idx, 1], 0.0, 0.0,
                                            self.cfg.DANGER_DELAY, 1.0, 0, 0, 0])
                else:
                    angle = np.random.uniform(0, 2 * np.pi)
                    dir_x, dir_y = np.cos(angle), np.sin(angle)
                    half_angle_deg = np.random.uniform(15, 30)
                    spread_cos = np.cos(np.radians(half_angle_deg))
                    self.danger = np.array(
                        [1.0, 2.0, self.boss[0], self.boss[1], dir_x, dir_y, self.cfg.BOSS_RADIUS, spread_cos, 0, 0, 0])

                self.boss[3] = 1.0
                self.boss[4] = self.cfg.DANGER_DELAY if atk_type == 1.0 else 0.5

        elif boss_state == 1.0:
            self.boss[4] -= self.cfg.DT
            if self.boss[4] <= 0:
                self.boss[3] = 2.0
                self.boss[4] = self.cfg.BOSS_CD_MAX

        elif boss_state == 2.0:
            self.boss[4] -= self.cfg.DT
            if self.boss[4] <= 0:
                self.boss[3] = 0.0

        if self.danger[0] == 1.0:
            if self.danger[1] == 1.0:
                self.danger[6] -= self.cfg.DT
                if self.danger[6] <= 0:
                    hit_someone = False
                    for i in range(self.cfg.NUM_AGENTS):
                        if self.agents[i, 4] == 1: continue
                        dist = np.linalg.norm(self.agents[i, :2] - self.danger[2:4])
                        if dist <= self.cfg.AOE_RADIUS:
                            self.agents[i, 2] -= self.cfg.DANGER_DMG
                            hit_someone = True
                            if self.agents[i, 2] <= 0: self.agents[i, 4] = 1.0
                    if not hit_someone: aoe_missed = True
                    self.danger[0] = 0.0

            elif self.danger[1] == 2.0:
                self.danger[6] += self.cfg.WAVE_SPEED * self.cfg.DT
                current_radius = self.danger[6]
                spread_cos = self.danger[7]

                for i in range(self.cfg.NUM_AGENTS):
                    if self.agents[i, 4] == 1 or self.danger[8 + i] == 1.0: continue
                    vec = self.agents[i, :2] - self.danger[2:4]
                    dist = np.linalg.norm(vec)

                    if abs(dist - current_radius) <= self.cfg.AGENT_RADIUS + self.cfg.WAVE_THICKNESS:
                        if dist > 0: vec /= dist
                        if np.dot(vec, self.danger[4:6]) >= spread_cos:
                            self.agents[i, 2] -= self.cfg.DANGER_DMG
                            self.danger[8 + i] = 1.0
                            if self.agents[i, 2] <= 0: self.agents[i, 4] = 1.0

                if current_radius >= self.cfg.WAVE_MAX_RADIUS:
                    if np.sum(self.danger[8:11]) == 0: aoe_missed = True
                    self.danger[0] = 0.0

        return aoe_missed

    def _get_boss_dists(self):
        dists = np.linalg.norm(self.agents[:, :2] - self.boss[:2], axis=1)
        return np.maximum(0, dists - self.cfg.BOSS_RADIUS - self.cfg.AGENT_RADIUS)

    def _get_obs(self):
        obs_dict = {}
        max_dist = np.sqrt(2) * self.cfg.MAP_SIZE

        for i in range(self.cfg.NUM_AGENTS):
            if self.agents[i, 4] == 1:
                obs_dict[f"agent_{i}"] = np.zeros(self.observation_space[f"agent_{i}"].shape, dtype=np.float32)
                continue

            obs = []
            obs.extend([
                self.agents[i, 0] / self.cfg.MAP_SIZE, self.agents[i, 1] / self.cfg.MAP_SIZE,
                self.agents[i, 2] / self.cfg.AGENT_MAX_HP, self.agents[i, 3] / self.cfg.AGENT_CD_MAX,
                self.agents[i, 5]
            ])

            boss_rel = self.boss[:2] - self.agents[i, :2]
            obs.extend([
                boss_rel[0] / self.cfg.MAP_SIZE, boss_rel[1] / self.cfg.MAP_SIZE,
                np.linalg.norm(boss_rel) / max_dist, max(0, self.boss[2]) / self.cfg.BOSS_MAX_HP,
                1.0 if self.boss[3] == 1.0 else 0.0, 1.0 if self.boss[5] == i else 0.0,
                self.boss[6], self.boss[7]
            ])

            if self.danger[0] == 1.0:
                d_rel = self.danger[2:4] - self.agents[i, :2]
                progress_ratio = self.danger[6] / self.cfg.DANGER_DELAY if self.danger[1] == 1.0 else self.danger[
                                                                                                          6] / self.cfg.WAVE_MAX_RADIUS
                obs.extend([
                    1.0, 1.0 if self.danger[1] == 1.0 else 0.0, 1.0 if self.danger[1] == 2.0 else 0.0,
                    d_rel[0] / self.cfg.MAP_SIZE, d_rel[1] / self.cfg.MAP_SIZE,
                    self.danger[4], self.danger[5], self.danger[7], progress_ratio
                ])
            else:
                obs.extend([0.0] * 9)

            for j in range(self.cfg.NUM_AGENTS):
                if i == j: continue
                if self.agents[j, 4] == 1:
                    obs.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    mate_rel = self.agents[j, :2] - self.agents[i, :2]
                    obs.extend([
                        mate_rel[0] / self.cfg.MAP_SIZE, mate_rel[1] / self.cfg.MAP_SIZE,
                        self.agents[j, 2] / self.cfg.AGENT_MAX_HP, 1.0
                    ])

            obs_dict[f"agent_{i}"] = np.array(obs, dtype=np.float32)

        return obs_dict