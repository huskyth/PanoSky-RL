from enum import Enum


class TrackStateEnum(Enum):
    NORMAL = 0  # 初始状态
    ADJUST_BOARD = 1
    CAPTURE = 2
    FIRE = 3
    LOAD_BULLET = 4


STATE_TO_STRING = {
    TrackStateEnum.NORMAL: "无目标",
    TrackStateEnum.ADJUST_BOARD: "调舷",
    TrackStateEnum.CAPTURE: "捕获",
    TrackStateEnum.FIRE: "开火",
    TrackStateEnum.LOAD_BULLET: "装弹",
}
