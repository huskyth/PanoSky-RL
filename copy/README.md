# 🚀 MAPPO-UAV-MicroBattle-Env

> 基于 MAPPO (Multi-Agent Proximal Policy Optimization) 深度强化学习的多智能体对抗小游戏仿真。
> 本项目构建了一个轻量级、高度自定义的 2D `Gymnasium` 连续控制物理环境。模拟了 3 台敏捷无人机（智能体）与 1 台重装目标（Boss）的非对称博弈。环境内置了物理碰撞、延迟弹道追踪、动态扇形扫射、毒雾隔离以及大招充能机制。

## ✨ 核心特性 (Features)

* 🌍 **二维连续物理引擎**：构建 1000 x 1000 的连续空间，支持基于弹性运动学的刚体防重叠隔离（Hard Collision）。
* ⚔️ **非对称动态博弈机制**：Boss 具备动态概率护盾、基于距离的毒雾光环（Poison Aura），以及全向随机角度与宽度的动态冲击波（Dynamic Wave），提升了环境的随机性与避障难度。
* 🚀 **延迟弹道与过载充能系统**：智能体攻击采用基于时间戳队列的延迟导弹模型（Missile Queue）。智能体通过存活积攒充能，满充能可释放无视护盾的三倍伤害过载打击，引导网络学习“延迟满足”与“生存优先”。
* 🎯 **方向性伤害判定与仇恨系统**：内置基于余弦相似度的正面装甲判定（Frontal Armor）。Boss 永远面朝仇恨目标，正面受击伤害减半，隐式驱使智能体学习诱饵拉扯与侧背绕后的多机协同。
* 🛡️ **反“摸鱼”与防“自杀”奖励重构**：引入独立计算的动态摸鱼惩罚（Anti-Slacking Penalty）与高额受伤惩罚，结合严格的截断机制，彻底杜绝多智能体博弈中常见的局部最优陷阱。

## ⚙️ 战斗博弈与物理引擎建模 (Combat & Physics Modeling)

本环境的核心难点在于多维度的动态博弈与时间延迟反馈。仿真步长精细至 0.1s。

**1. 延迟弹道与护盾结算 (Delayed Missile & Shield)**
当智能体在 $t$ 时刻进入射程开火时，基于当前距离 $d$ 计算导弹飞行时间 $\Delta t = d / V_{missile}$，将伤害推入时间戳队列。Boss 在受击瞬间，根据当前血量比例 $HP_{ratio}$ 拥有动态免疫概率 $P_{immune}$：

$$P_{immune} = P_{base} + (1 - HP_{ratio}) \times (P_{max} - P_{base})$$

**2. 动态扇形扫射 (Dynamic Sector Wave)**
Boss 发射的冲击波并非静态区域，而是随时间 $t$ 扩散的波面。波面半径如下：

$$R_{wave}(t) = V_{wave} \times t$$

对于任一智能体，仅当其与危险源距离 $d \approx R_{wave}$，且其相对方向向量 $\mathbf{v}$ 与冲击波中心方向 $\mathbf{u}$ 的点积满足 $\mathbf{v} \cdot \mathbf{u} \ge \cos(\theta_{spread})$ 时，触发碰撞伤害。

**3. 毒雾光环与防穿模机制 (Poison Aura & Hard Collision)**
为防止智能体通过贴近 Boss 使得弹道延迟 $\Delta t \to 0$ 从而刷取伤害，系统设定距离 $d < R_{poison}$ 为毒雾区，施加高额秒伤。同时底层物理运算在每步执行向量归一化斥力，绝对禁止实体间距小于半径之和：

$$\mathbf{p}_i = \mathbf{p}_{boss} + \frac{\mathbf{p}_i - \mathbf{p}_{boss}}{\|\mathbf{p}_i - \mathbf{p}_{boss}\|} \times (R_i + R_{boss} + \epsilon)$$

## 🧠 强化学习环境设计 (MDP Design)

项目通过严密的数值缩放与奖励平衡，实现端到端控制。

### 1. 状态空间 (Observation Space)

状态空间采用相对物理量，并经过严格的 `[-1.0, 1.0]` 或 `[0.0, 1.0]` 归一化处理。单智能体局部观测维度为 **30 维**：

| 维度索引 | 物理含义 (Description) | 说明 |
| :---: | :--- | :--- |
| `0:5` | **智能体自身状态** | 归一化坐标 `(x, y)`，血量比例，攻击 CD，大招充能进度 |
| `5:13` | **Boss 相对状态** | 相对向量 `(Δx, Δy)`，相对距离，血量，施法前摇，仇恨目标指示，朝向向量 |
| `13:22` | **动态威胁区 (Danger Zone)** | 激活状态，类型(红圈/冲击波)，相对向量，波面朝向，波面宽度余弦值，传播进度比例 |
| `22:30` | **队友相对状态** | 遍历 2 名队友的相对向量 `(Δx, Δy)`，血量比例，存活状态 |

### 2. 动作空间 (Action Space)

采用 **2 维连续动作空间**，控制智能体在 2D 平面内的期望速度分量：
* $\mathbf{a}_t \in [-1.0, 1.0]^2$：网络输出经过 Tanh 激活，实际指令速度公式为：

$$\mathbf{v}_{actual} = \mathbf{a}_t \times V_{max}$$

### 3. 奖励函数设计 (Reward Function)

单步奖励受限于严格的 `[-2.0, 2.0]` 截断机制（稀疏事件奖励除外）：

$$R_t = r_{step} + r_{nav} + r_{danger} + r_{surround} + r_{dmg} + r_{slack} + r_{hurt}$$

* 🟢 **距离势能与包围 ($r_{nav}, r_{surround}$)**：在安全作战半径内，根据 3 名智能体相对 Boss 方向向量的和的模长计算包围质量，引导形成 120 度交叉火力。
* 🟡 **动态避障与摸鱼惩罚 ($r_{danger}, r_{slack}$)**：当剑气逼近时，依据距离提供指数级陡峭的负梯度，逼迫智能体切向移动。针对未被仇恨锁定且游离于战场外的智能体，施加基于距离的固定摸鱼惩罚。
* 🔴 **伤害与承伤机制 ($r_{dmg}, r_{hurt}$)**：严密控制 $r_{hurt}$ 绝对值大于基础 $r_{dmg}$，彻底打破“以血换输出”的局部最优解。仅当智能体利用背刺乘区或过载大招时，收益才能取得正向突破。


## 🛠️ 环境依赖与快速开始 (Quick Start)

推荐环境：Python 3.8+ / PyTorch 2.0+

```bash
# 安装基础依赖库
pip install gymnasium torch numpy pygame pandas matplotlib imageio
```

![battle_record_ep1](https://github.com/user-attachments/assets/b5420b4d-51fc-47e3-b0ab-ae6f2710f6eb)
