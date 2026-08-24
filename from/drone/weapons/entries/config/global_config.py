UNIT_TIME = 0.1  # 时间单位
TIME_TO_CONFIRM_A_UAV = 3  # 搜索确认一架飞机的次数
CAPTURE_TIME = 4  # 捕获时间
FIRE_INSTRUCTION_TIME = 0.1  # 开火指令耗时
REMAIN_TIME_TO_ADJUST_BOARD_IN_CAPTURE_STATE = 5  # 捕获状态剩下多久开始去调舷

from enum import Enum


class ThreatLevelPriority(Enum):
    VELOCITY = 0
    DISTANCE = 1


class GameConfig:
    CURRENT_GAME_TIME = 0
    threat_level_priority = ThreatLevelPriority.DISTANCE
    record_unity_end_flag = 0
    TEST_TURN = 0
    RANDOM_N = 0
    THREAT_LEVEL_THRESHOLD = 0  # 威胁程度阈值

    @staticmethod
    def reset():
        GameConfig.TEST_TURN = 0
        GameConfig.RANDOM_N = 0

        GameConfig.THREAT_LEVEL_THRESHOLD = 0  # 威胁程度阈值
        GameConfig.CURRENT_GAME_TIME = 0
        GameConfig.map = None
        GameConfig.threat_level_priority = ThreatLevelPriority.DISTANCE
        GameConfig.record_unity_end_flag = 0
