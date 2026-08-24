from onpolicy.envs.drone.weapons.factory.config_factory import ConfigFactory
from onpolicy.envs.drone.weapons.factory.phalanx_factory import PhalanxFactory
from onpolicy.envs.drone.weapons.factory.uav_factory import UAVFactory
from onpolicy.envs.drone.weapons.entries.phalanx.components.track_rader.track_rader_state import *
from onpolicy.envs.drone.weapons.entries.config.global_config import *
from onpolicy.utils.format_logger import _green_log_str
from onpolicy.utils.math_tool import distance_of_2_point, length_of_vector
from onpolicy.utils.util import compute_distance


class GameManager:
    uav_list = None
    phalanx = None
    search_rader, track_rader, weapon = None, None, None

    @staticmethod
    def get_uav_index(uav_param=None):
        if uav_param:
            for i, uav in enumerate(GameManager.uav_list):
                if uav_param is uav:
                    return str(i)
        else:
            for i, uav in enumerate(GameManager.uav_list):
                if GameManager.track_rader.get_current_target() is uav:
                    return str(i)
        assert False, "can not find uav"

    @staticmethod
    def instance(*keys, **kwards):
        if GameManager.phalanx is None:
            GameManager.phalanx = PhalanxFactory.create(*keys, **kwards)
        GameManager.search_rader, GameManager.track_rader, GameManager.weapon = GameManager.phalanx.search_rader, GameManager.phalanx.track_rader, GameManager.phalanx.weapon
        GameManager.uav_list = GameManager._create_uav(keys[0])

    @staticmethod
    def print_singletance():
        print("GameManager.phalanx:" + str(GameManager.phalanx) + "\nGameManager.search_rader:" + str(
            GameManager.search_rader) + "\nGameManager.track_rader" + str(
            GameManager.track_rader) + "\nGameManager.weapon:" + str(
            GameManager.weapon))

    @staticmethod
    def _create_uav(num):
        return [UAVFactory.create() for i in range(num)]

    @staticmethod
    def generate_no_none_list():
        return [x for x in GameManager.uav_list if x is not None]

    @staticmethod
    def is_first_uav_exist():
        return GameManager.uav_list[0] is not None

    @staticmethod
    def is_uav_exist():
        return len(GameManager.generate_no_none_list()) > 0

    @staticmethod
    def uav_position_update(position):
        for i, a_uav in enumerate(GameManager.uav_list):
            if a_uav:
                a_uav.position = position[i].copy()

    @staticmethod
    def set_weapon_position(position):
        GameManager.phalanx.set_position(position)

    @staticmethod
    def uav_velocity_update(actions):
        '''
        @param actions: [(方向),...]
        '''
        for idx, v in enumerate(actions):
            assert idx <= len(GameManager.uav_list) - 1, "action length less than length of uav list"
            # GameManager.uav_list[idx].velocity = distance_of_2_point(v, v)
            if GameManager.uav_list[idx]:
                GameManager.uav_list[idx].velocity = length_of_vector(v)
                GameManager.uav_list[idx].velocity_direction = v.copy()

    @staticmethod
    def uav_position_reset(positions):
        assert len(positions) > 0, "uav positions length < 0"
        assert len(positions) == len(GameManager.uav_list), 'length not equal'
        for idx, s_p in enumerate(positions):
            GameManager.uav_list[idx].position = s_p.copy()

    @staticmethod
    def get_uav_list():
        return GameManager.uav_list

    @staticmethod
    def get_phalanx():
        return GameManager.phalanx

    @staticmethod
    def get_current_state():
        return TrackRaderState.get_current_string_state()

    @staticmethod
    def reset():
        GameManager.uav_list = None
        GameManager.phalanx = None
        GameManager.search_rader, GameManager.track_rader, GameManager.weapon = None, None, None

    @staticmethod
    def destroy():
        GameManager.reset()
        GameConfig.reset()
        TrackRaderState.reset()
        ConfigFactory.reset()
        PhalanxFactory.reset()
        return


class AbstractAction:
    @staticmethod
    def execute():
        pass


class UnityTestAction(AbstractAction):
    one_print_flag = False
    adjust_num = 0

    def write_info(self):
        if len(GameManager.uav_list) != 0:
            logger.info("UAV:" + ",".join([str(x) for x in GameManager.uav_list[0].get_position()]))
        logger.info("P_TOP:" + ",".join([str(x) for x in GameManager.track_rader.top_board_position_vector]))

    def execute(self):
        if not UnityTestAction.one_print_flag:
            logger.info("P:" + ",".join([str(x) for x in GameManager.phalanx.get_position()]))
            UnityTestAction.one_print_flag = True
        if GameConfig.record_unity_end_flag != 1:
            self.write_info()
            if GameConfig.record_unity_end_flag == 0.5:
                GameConfig.record_unity_end_flag = 1


class TimeStepAction(AbstractAction):
    @staticmethod
    def execute():
        content = TrackRaderState.get_current_string_state()
        gre = _green_log_str(f"{'=' * 123}")
        logger.info(f"✅【当前时间】{GameConfig.CURRENT_GAME_TIME}【当前状态为】-{content} {gre}")
        logger.info("当前时间{},随机数n为{}".format(GameConfig.CURRENT_GAME_TIME, GameConfig.RANDOM_N))
        GameConfig.CURRENT_GAME_TIME = round(UNIT_TIME + GameConfig.CURRENT_GAME_TIME, 1)
        for i in range(len(GameManager.uav_list)):
            if not GameManager.uav_list[i]:
                temp = None
            else:
                temp = GameManager.uav_list[i].position

            temp_p = GameManager.track_rader.position
            logger.info(f"当前无人机位置为{temp}, 距离为 {'None' if not temp else compute_distance(temp_p, temp)}",
                        is_in_file=False)
        if GameManager.track_rader.current_target is not None:
            logger.info("当前目标索引为{}".format(GameManager.get_uav_index()))


class SearchAction(AbstractAction):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _search_log():
        print_uav_list = GameManager.search_rader.get_mark_3_time_uav_list(GameManager.uav_list)
        uav_list_str = [uav.print_self() for uav in print_uav_list]
        logger.info("搜索雷达发现的目标", uav_list_str)
        content = "不存在" if len(uav_list_str) == 0 else " ".join(uav_list_str)

        logger.info("搜索雷达发现的目标：" + content)

    @staticmethod
    def execute():
        # 每一轮都执行
        SearchAction._search_log()
        GameManager.search_rader.search_object(GameManager.uav_list)


class ConfirmTargetAction(AbstractAction):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _target_log():
        current_target = GameManager.track_rader.try_get_current_target()
        content = "不存在" if current_target is None else current_target.print_self()
        logger.info("当前目标", [content])
        logger.info("当前目标为：" + content)

    @staticmethod
    def execute():
        if TrackRaderState.is_normal_state():
            if GameManager.track_rader.confirm_track_target(GameManager.search_rader,
                                                            GameManager.uav_list,
                                                            GameManager.get_uav_index) is not None:
                TrackRaderState.set_adjust_board_state()

                GameManager.track_rader.calculate_adjust_data()

        ConfirmTargetAction._target_log()


class AdjustBoardAction(AbstractAction):

    def __init__(self):
        super().__init__()

    def execute(self):
        if TrackRaderState.is_adjust_board_state():
            # 当前目标存在，只在调舷状态执行调舷
            is_in_track_range = GameManager.track_rader.is_target_in_track_range(GameManager.get_uav_index)
            can_use_because_no_mountain = GameManager.track_rader.is_target_can_use_because_no_mountain()
            if not can_use_because_no_mountain:
                logger.info(f"会因为不能用而被移除")
            if not is_in_track_range:
                GameManager.track_rader.remove_target(GameManager.weapon)
                GameManager.weapon.reset_fire_instruction()
                TrackRaderState.set_normal_state()
                GameManager.track_rader.adjust_end_set_value()
            else:
                AdjustBoardAction._adjust_board()

    @staticmethod
    def _adjust_board():
        can_capture = GameManager.track_rader.adjust_board(GameManager.get_uav_index)
        if can_capture:
            if GameConfig.record_unity_end_flag == 0:
                GameConfig.record_unity_end_flag = 0.5
            TrackRaderState.set_capture_state()
        else:
            # 不能捕获依旧调舷状态
            pass


class CaptureAction(AbstractAction):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _capture():
        GameManager.track_rader.step_capture()
        if GameManager.track_rader.check_can_fire():
            TrackRaderState.set_fire_state()
        else:
            # 不能开火，维持原始
            pass

    @staticmethod
    def execute():
        if TrackRaderState.is_capture_state():
            is_in_track_range = GameManager.track_rader.is_target_in_track_range(GameManager.get_uav_index)
            can_use_because_no_mountain = GameManager.track_rader.is_target_can_use_because_no_mountain()
            if not can_use_because_no_mountain:
                logger.info(f"会因为不能用而被移除")
            if not is_in_track_range:
                GameManager.track_rader.remove_target(GameManager.weapon)
                GameManager.track_rader.reset_capture_time()
                GameManager.weapon.reset_fire_instruction()
                TrackRaderState.set_normal_state()
            else:
                CaptureAction._capture()


class LoadBulletAction(AbstractAction):
    end_state = None

    def __init__(self):
        super().__init__()

    @staticmethod
    def confirm_track_target_and_adjust_board():
        if GameManager.track_rader.confirm_track_target(GameManager.search_rader, GameManager.uav_list,
                                                        GameManager.get_uav_index) is not None:
            if LoadBulletAction.end_state != TrackStateEnum.ADJUST_BOARD:
                GameManager.track_rader.calculate_adjust_data()
                LoadBulletAction.end_state = TrackStateEnum.ADJUST_BOARD
            if GameManager.track_rader.adjust_board(GameManager.get_uav_index):
                LoadBulletAction.end_state = TrackStateEnum.CAPTURE
            else:
                LoadBulletAction.end_state = TrackStateEnum.ADJUST_BOARD
        else:
            LoadBulletAction.end_state = TrackStateEnum.NORMAL

    @staticmethod
    def _adjust_board_in_load_bullet_action():
        if GameManager.track_rader.try_get_current_target() is None:
            logger.info("装弹的时候调弦，但是目标已经移除")
            LoadBulletAction.confirm_track_target_and_adjust_board()
            return
        is_target_in_track_range = GameManager.track_rader.is_target_in_track_range(GameManager.get_uav_index)
        can_use = GameManager.track_rader.is_target_can_use_because_no_mountain()
        if not can_use:
            logger.info(f"会因为不能用而被移除")
        if not is_target_in_track_range:
            LoadBulletAction.end_state = TrackStateEnum.NORMAL
            GameManager.track_rader.remove_target(GameManager.weapon)
            GameManager.weapon.reset_fire_instruction()
            LoadBulletAction.confirm_track_target_and_adjust_board()
            GameManager.track_rader.adjust_end_set_value()
        else:
            # 还在范围内,直接返回
            LoadBulletAction.end_state = TrackStateEnum.FIRE

    @staticmethod
    def execute():
        LoadBulletAction.end_state = None
        # 装弹是否成功
        if TrackRaderState.is_load_bullet_state():
            LoadBulletAction._adjust_board_in_load_bullet_action()
            if GameManager.weapon.step_load_bullet():
                logger.info("装弹完成，末状态为：" + STATE_TO_STRING[LoadBulletAction.end_state])
                # 装弹完成,万一没走？
                TrackRaderState.set_definded_state(LoadBulletAction.end_state)
            else:
                # 下次继续装弹
                pass


class BulletAttackAction(AbstractAction):
    def __init__(self):
        super().__init__()

    @staticmethod
    def execute():
        will_deleted_fired_bullet_list = []

        for idx, a_bullet in enumerate(GameManager.weapon.fired_bullet_list):
            # 列表中的子弹代表发射出去的，默认都有跟踪目标
            bullet_state = a_bullet.step_attack_a_target_and_is_kill(
                GameManager.uav_list, GameManager.get_uav_index)
            if bullet_state == GameManager.weapon.BulletState.KILLED_NO_USE:
                GameManager.track_rader.remove_target(GameManager.weapon)
                GameManager.weapon.reset_fire_instruction()
                TrackRaderState.set_normal_state()
                GameManager.weapon.clear_fired_bullet_list()
                return
            else:
                pass
            if bullet_state != GameManager.weapon.BulletState.FLYING_USEING:
                will_deleted_fired_bullet_list.append(idx)

        for will_delete_a_bullet in will_deleted_fired_bullet_list[::-1]:
            del GameManager.weapon.fired_bullet_list[will_delete_a_bullet]


class FireAction(AbstractAction):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _fire():
        # 这里开火状态决定了可以开火，而不是>0 或者满子弹状态混淆，所以以下子弹>0就可以开火
        bullet_exist = GameManager.weapon.is_bullet_can_fire()
        if bullet_exist:
            if GameManager.weapon.is_fire_instruction_finish():
                GameManager.weapon.fire(GameManager.track_rader.get_current_target())
            else:
                GameManager.weapon.step_time_instruction()
        else:
            TrackRaderState.set_bullet_load_state()

    @staticmethod
    def execute():
        if TrackRaderState.is_fire_state():
            is_target_in_fire_range = GameManager.track_rader.is_target_in_fire_range()
            logger.info(
                "最小开火距离为：" + str(GameManager.track_rader.minimum_fire_distance) + "，开火最远距离为：" + str(
                    GameManager.track_rader.fire_distance) + "，密集阵与id为" + str(
                    GameManager.get_uav_index()) + "的无人机的距离：" + str(
                    distance_of_2_point(GameManager.track_rader.position,
                                        GameManager.track_rader.current_target.position)), is_in_file=False)
            can_use = GameManager.track_rader.is_target_can_use_because_no_mountain()
            if not can_use:
                logger.info(f"会因为不能用而被移除")
            if not is_target_in_fire_range:
                logger.info("因为超出开火距离或者有山而被设置为没有目标状态")
                GameManager.track_rader.remove_target(GameManager.weapon)
                GameManager.weapon.reset_fire_instruction()
                TrackRaderState.set_normal_state()
            else:
                FireAction._fire()
