from onpolicy.envs.drone.weapons.entries.phalanx.components.track_rader.track_rader_enum import *
from onpolicy.utils.format_logger import AppLogger

logger = AppLogger().get_logger()


def print_wrapper(to_state):
    def wrapper(func):
        def print_state():
            logger.info("状态转换过程：" + str(
                STATE_TO_STRING[TrackRaderState.current_state]) + "->" + str(
                to_state))
            func()

        return print_state

    return wrapper


class TrackRaderState:
    current_state = TrackStateEnum.NORMAL

    @staticmethod
    def get_current_string_state():
        return STATE_TO_STRING[TrackRaderState.current_state]

    @staticmethod
    def get_current_int_state():
        return TrackRaderState.current_state.value

    @staticmethod
    def is_adjust_board_state():
        return TrackRaderState.current_state == TrackStateEnum.ADJUST_BOARD

    @staticmethod
    def is_normal_state():
        return TrackRaderState.current_state == TrackStateEnum.NORMAL

    @staticmethod
    def is_capture_state():
        return TrackRaderState.current_state == TrackStateEnum.CAPTURE

    @staticmethod
    def is_fire_state():
        return TrackRaderState.current_state == TrackStateEnum.FIRE

    @staticmethod
    def is_load_bullet_state():
        return TrackRaderState.current_state == TrackStateEnum.LOAD_BULLET

    @staticmethod
    @print_wrapper(STATE_TO_STRING[TrackStateEnum.FIRE])
    def set_fire_state():
        logger.info(f"状态转化过程：{TrackRaderState.current_state} -> {TrackStateEnum.FIRE}")
        TrackRaderState.current_state = TrackStateEnum.FIRE

    @staticmethod
    @print_wrapper(STATE_TO_STRING[TrackStateEnum.NORMAL])
    def set_normal_state():
        TrackRaderState.current_state = TrackStateEnum.NORMAL

    @staticmethod
    @print_wrapper(STATE_TO_STRING[TrackStateEnum.LOAD_BULLET])
    def set_bullet_load_state():
        TrackRaderState.current_state = TrackStateEnum.LOAD_BULLET

    @staticmethod
    @print_wrapper(STATE_TO_STRING[TrackStateEnum.ADJUST_BOARD])
    def set_adjust_board_state():
        TrackRaderState.current_state = TrackStateEnum.ADJUST_BOARD

    @staticmethod
    @print_wrapper(STATE_TO_STRING[TrackStateEnum.CAPTURE])
    def set_capture_state():
        TrackRaderState.current_state = TrackStateEnum.CAPTURE

    STATE_TO_FUNCTION = {
        TrackStateEnum.NORMAL: set_normal_state,
        TrackStateEnum.ADJUST_BOARD: set_adjust_board_state,
        TrackStateEnum.CAPTURE: set_capture_state,
        TrackStateEnum.FIRE: set_fire_state,
        TrackStateEnum.LOAD_BULLET: set_bullet_load_state,
    }

    @staticmethod
    def set_definded_state(state_value):
        if state_value == TrackStateEnum.NORMAL:
            TrackRaderState.set_normal_state()
        elif state_value == TrackStateEnum.ADJUST_BOARD:
            TrackRaderState.set_adjust_board_state()
        elif state_value == TrackStateEnum.CAPTURE:
            TrackRaderState.set_capture_state()
        elif state_value == TrackStateEnum.FIRE:
            TrackRaderState.set_fire_state()
        elif state_value == TrackStateEnum.LOAD_BULLET:
            TrackRaderState.set_bullet_load_state()
        else:
            assert state_value in TrackRaderState.STATE_TO_FUNCTION, "state not in state values"

    @staticmethod
    def reset():
        TrackRaderState.current_state = TrackStateEnum.NORMAL
