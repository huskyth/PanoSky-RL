import configparser
import json
import os
import time

from onpolicy.envs.drone.pages.mountain_add import write_mdata_to_file
from onpolicy.utils.util import compute_distance
from weapon_target import write_data_to_file

# ===================== 配置参数 =====================
# 数据文件路径
DATA_FILE = "drone_data.json"
# 每一步模拟间隔(秒)
STEP_INTERVAL = 1
# 一轮路径跑完后停顿时间(秒)
LOOP_PAUSE = 3.0

from pathlib import Path

temp = Path(__file__).parent.parent

config_path = temp / 'config/th_demo.ini'
cf = configparser.ConfigParser()
cf.read(str(config_path), encoding="utf-8")
MOUNTAIN_NAME = cf.get("map", "map_data_file")

rm_file = str(temp / f"pages/{DATA_FILE}")
if os.path.exists(rm_file):
    os.remove(rm_file)


# ===================== 1. 一次性读取 目标/武器 数据 =====================
def load_target_weapon():
    """读取target和weapon，只加载一次"""
    try:
        with open("sim_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        target = config.get("target", [0, 0, 0])
        weapon = config.get("weapon", [0, 0, 0])
        print(f"✅ 读取目标点: {target}")
        print(f"✅ 读取武器点: {weapon}")
        return target, weapon
    except Exception as e:
        print(f"❌ 读取配置失败: {e}")
        return [0, 0, 0], [0, 0, 0]


# ===================== 2. 读取无人机路径序列 =====================
def load_episode_path():
    """读取完整的episode路径数据"""
    try:
        with open("sim_config.json", "r", encoding="utf-8") as f:
            path_data = json.load(f)
        episode_data = path_data.get("episode_data", [])
        print(f"✅ 读取路径总步数: {len(episode_data)}")
        return episode_data
    except Exception as e:
        print(f"❌ 读取路径失败: {e}")
        return []


# ===================== 3. 主循环：循环执行完整路径 =====================
def main():
    time.sleep(5)
    map = write_mdata_to_file(MOUNTAIN_NAME)
    # 一次性加载目标、武器
    target_pos, weapon_pos = load_target_weapon()
    wx, wy, wz = weapon_pos
    tx, ty, tz = target_pos
    write_data_to_file(*map.for_html(wx, wy), wz / 1, *map.for_html(tx, ty), tz / 1)
    # 加载完整路径
    episode_path = load_episode_path()

    if not episode_path:
        print("❌ 无路径数据，退出")
        return

    while True:
        print("\n🚀 开始新一轮路径飞行...")
        # 逐步执行路径
        for step_info in episode_path:
            uva_state_list = step_info.get("uva_state", [])
            step = step_info.get("_episode_steps", 0)
            # assert len(uva_state_list) == 2
            # 构造输出给前端的无人机数据
            drones = []
            reward = step_info['reward']
            r_msg = step_info.get('r_msg', "Not set")
            deg = step_info.get('degree', "Not set")
            print(f"📌 == 【Step】{step}" + '=' * 150)
            r_msg = '\n'.join(r_msg)
            deg = [str(f) for f in deg]
            deg = '\n'.join(deg)
            print(f"当前目标为 {step_info['c_target_id']},  各自奖励为 {reward}，奖励描述为: \n {r_msg}")
            print(f'威胁程度为 {deg}')

            pos_0 = None
            dis = None
            for idx, uva in enumerate(uva_state_list):
                pos = uva.get("position", [0, 0, 0])
                if idx == 0:
                    pos_0 = pos
                else:
                    dis = compute_distance(pos_0, pos)
                print(
                    f"ID: {uva['ID']}-【{uva.get('status', 'None')}】-【无人机与目标的真实距离】 {compute_distance(pos, target_pos)}，"
                    f"【无人机高度】{pos[2]}， 【无人机与武器的真实距离】 {compute_distance(pos, weapon_pos)}"
                    f"")
                x, y = map.for_html(pos[0], pos[1])
                drones.append({
                    "id": f"UAV-{idx + 1:03d}",
                    "x": x,
                    "z": y,
                    "altitude": pos[2] / 1,
                    "speed": 11.414917009282588,
                    "battery": 97.85000000000012,
                    "aimX": map.for_html(wx, wy)[0],
                    "aimY": step * 2,
                    "aimZ": map.for_html(wx, wy)[1],
                })
            print(f"两机距离为 {dis}")
            # 写入JSON文件，供前端Three.js读取
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(drones, f, ensure_ascii=False, indent=2)

            print()

            time.sleep(STEP_INTERVAL)

        # 一轮路径跑完，停顿一段时间再循环
        print(f"⏸️  本轮路径结束，停顿 {LOOP_PAUSE}s 后重启...")
        time.sleep(LOOP_PAUSE)
        break


if __name__ == "__main__":
    main()
