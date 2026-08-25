import json
import math


def generate_coordinated_trajectory(steps=70):
    data = []
    weapon_pos = [-15.0, 0.0, -20.0]
    target_pos = [0.0, 0.0, 0.0]

    bait_start = [-20.0, 0.0, 20.0]  # 诱饵起点（左侧）
    assassin_start = [-30.0, 0.0, -30.0]  # 刺客起点（右后侧）

    for step in range(steps):
        t = step / steps

        # 诱饵：从起点沿弧线向目标靠近，同时横向摆动（模拟机动）
        bait_x = bait_start[0] + (target_pos[0] - bait_start[0]) * t + 8 * math.sin(step * 0.12)
        bait_z = bait_start[2] + (target_pos[2] - bait_start[2]) * t - 6 * (1 - math.cos(step * 0.08))
        bait_pos = [bait_x, 0.0, bait_z]

        # 刺客：直线加速冲向目标，但稍慢于诱饵
        ass_t = min(1.0, t * 1.2)  # 刺客稍晚到达，体现协同
        ass_x = assassin_start[0] + (target_pos[0] - assassin_start[0]) * ass_t
        ass_z = assassin_start[2] + (target_pos[2] - assassin_start[2]) * ass_t
        ass_pos = [ass_x, 0.0, ass_z]

        # 武器状态基于诱饵距离
        dist = math.sqrt((bait_x - weapon_pos[0]) ** 2 + (bait_z - weapon_pos[2]) ** 2)
        if dist < 18:
            state = 3  # FIRE
            tuning_time = 0.0
            # 子弹从武器位置射向诱饵方向
            dx = bait_x - weapon_pos[0]
            dz = bait_z - weapon_pos[2]
            length = math.sqrt(dx ** 2 + dz ** 2)
            if length > 0:
                bullets = [{"x": weapon_pos[0] + dx / length * 10,
                            "y": 0.0,
                            "z": weapon_pos[2] + dz / length * 10}]
            else:
                bullets = []
        elif dist < 28:
            state = 2  # CAPTURE
            tuning_time = 0.0
            bullets = []
        elif dist < 38:
            state = 1  # TUNING
            tuning_time = 0.4 - (dist - 28) / 10 * 0.4
            bullets = []
        else:
            state = 0  # NORMAL
            tuning_time = 0.0
            bullets = []

        # 瞄准线指向诱饵
        aim_x, aim_y, aim_z = bait_x, 0.0, bait_z

        # 检查刺客是否到达目标
        ass_dist = math.sqrt(ass_pos[0] ** 2 + ass_pos[2] ** 2)
        if ass_dist < 2.0 and step > 15:
            reward = [100.0, 100.0]
            is_terminal = [True, True]
        else:
            reward = [0.0, 0.0]
            is_terminal = [False, False]

        uavs = [
            {
                "id": "UAV-000",
                "position": bait_pos,
                "velocity": [(target_pos[0] - bait_start[0]) / steps + 0.8 * math.cos(step * 0.12), 0.0,
                             (target_pos[2] - bait_start[2]) / steps + 0.8 * math.sin(step * 0.08)],
                "speed": 5.5,
                "battery": 100,
                "is_decoy": True,
                "is_targeted": (state >= 1)
            },
            {
                "id": "UAV-001",
                "position": ass_pos,
                "velocity": [(target_pos[0] - assassin_start[0]) / steps * 1.2, 0.0,
                             (target_pos[2] - assassin_start[2]) / steps * 1.2],
                "speed": 6.0,
                "battery": 100,
                "is_decoy": False,
                "is_targeted": False
            }
        ]

        weapon = {
            "id": "WPN-001",
            "entityType": "weapon",
            "position": weapon_pos,
            "state": state,
            "tuning_time": tuning_time,
            "aim_x": aim_x,
            "aim_y": aim_y,
            "aim_z": aim_z,
            "range": 70.0,
            "ammo": 1000
        }

        target = {
            "id": "TGT-001",
            "entityType": "target",
            "position": target_pos,
            "threatLevel": "高",
            "targetType": "雷达站",
            "threatRange": 50
        }

        step_record = {
            "step": step,
            "reward": reward,
            "is_terminal": is_terminal,
            "uavs": uavs,
            "weapon": weapon,
            "target": target,
            "bullets": bullets
        }
        data.append(step_record)

    return data


if __name__ == "__main__":
    trajectory = generate_coordinated_trajectory(80)
    with open("突防轨迹_协同.json", "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)
    print("✅ 协同突防轨迹已生成: 突防轨迹_协同.json")