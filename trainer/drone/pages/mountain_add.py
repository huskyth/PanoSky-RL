from onpolicy.envs.drone.maps.map import Map
import json


def write_mdata_to_file(mountain):
    map = Map(mountain, 30, 3.05)

    heights = (map.map_z / 1).flatten().tolist()

    # ========== 3. 地形数据（可选） ==========

    terrain_data = {"heights": heights}

    with open('terrain_data.json', 'w', encoding='utf-8') as f:
        json.dump(terrain_data, f, ensure_ascii=False, indent=2)

    print("✅ 数据文件已生成")
    print("🚀 请用浏览器打开 drone_monitor.html")
    return map

if __name__ == '__main__':
    write_mdata_to_file()