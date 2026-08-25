import json
import numpy as np
import time


def write_data_to_file(wx, wy, wz, tx, ty, tz):
    # ========== 2. 武器/目标数据（固定位置） ==========
    entity_data = [
        # 武器站
        {
            "id": "WPN-001",
            "entityType": "weapon",  # 必填：weapon 或 target
            "x": wx,
            "z": wy,
            "altitude": wz,
            "range": 60,  # 射程（米）
            "ammo": 120,  # 弹药量
            "azimuth": 45,  # 方位角
            "elevation": 15,  # 俯仰角
            "fireRadius": 1500
        },

        # 目标
        {
            "id": "TGT-001",
            "entityType": "target",  # 必填
            "x": tx,
            "z": ty,
            "altitude": tz,
            "threatLevel": "高",  # 高/中/低/极高
            "targetType": "装甲车",  # 自定义类型
            "threatRange": 40  # 威胁范围（米）
        },

    ]

    with open('entity_data.json', 'w', encoding='utf-8') as f:
        json.dump(entity_data, f, ensure_ascii=False, indent=2)
