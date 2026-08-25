import os

from onpolicy.envs.drone.weapons.factory.action_factory import *


class Game:
    action_init_bool = False
    uav_penetration_action = None
    search_action = None
    adjust_board_action = None
    capture_action = None
    load_bullet_action = None
    fire_action = None
    confirm_target_action = None
    bullet_attack_action = None
    time_step_action = None
    action_list = None

    @staticmethod
    def _action_init():
        Game.search_action = ActionFactory.create(ActionEnum.SearchAction)
        Game.adjust_board_action = ActionFactory.create(ActionEnum.AdjustBoardAction)
        Game.capture_action = ActionFactory.create(ActionEnum.CaptureAction)
        Game.load_bullet_action = ActionFactory.create(ActionEnum.LoadBulletAction)
        Game.fire_action = ActionFactory.create(ActionEnum.FireAction)
        Game.confirm_target_action = ActionFactory.create(ActionEnum.ConfirmTargetAction)
        Game.bullet_attack_action = ActionFactory.create(ActionEnum.BulletAttackAction)
        Game.time_step_action = ActionFactory.create(ActionEnum.TimeStepAction)
        Game.action_list = [Game.time_step_action, Game.search_action, Game.confirm_target_action,
                            Game.adjust_board_action,
                            Game.capture_action, Game.fire_action,
                            Game.load_bullet_action,
                            Game.bullet_attack_action]

    @staticmethod
    def reset(number_uav, weapon_positions, uav_velocity, uav_position, mmap):
        if not Game.action_init_bool:
            Game._action_init()
            Game.action_init_bool = True
        GameManager.destroy()
        GameManager.instance(number_uav, map=mmap)
        GameManager.phalanx.set_position(weapon_positions)
        GameManager.uav_velocity_update(uav_velocity)
        GameManager.uav_position_update(uav_position)

    @staticmethod
    def step(action, uav_velocity):
        GameManager.uav_position_update(action)
        GameManager.uav_velocity_update(uav_velocity)

        for action in Game.action_list:
            if isinstance(action, BulletAttackAction):
                action.execute()
            else:
                action.execute()
        logger.info(f"这里返回的UAV ID {[(id(x), x) for x in GameManager.get_uav_list()]}")
        return GameManager.get_uav_list()

    @staticmethod
    def try_get_current_target():

        return GameManager.track_rader.try_get_current_target()

    @staticmethod
    def get_uav_list():
        return GameManager.get_uav_list()

    @staticmethod
    def get_weapon_state():
        return TrackRaderState.get_current_int_state()