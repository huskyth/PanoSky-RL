from onpolicy.envs.drone.weapons.entries.config.global_config import GameConfig, UNIT_TIME
from onpolicy.envs.drone.weapons.factory.config_factory import ConfigFactory, ConfigEnum
from onpolicy.envs.drone.weapons.interfaces.game import Game


class EnvironmentInterface:
    @staticmethod
    def get_game_unit_time():
        return UNIT_TIME

    @staticmethod
    def get_rader_detect_radius():
        return ConfigFactory.create(config_type=ConfigEnum.phalanx).track_rader_config.get_track_distance()

    @staticmethod
    def get_fire_distance():
        return ConfigFactory.create(config_type=ConfigEnum.phalanx).track_rader_config.get_fire_distance()

    @staticmethod
    def step(actions, uav_velocity):
        return Game.step(actions, uav_velocity)

    @staticmethod
    def reset(number_uav, weapon_positions, uav_velocity, uav_position, mmap):
        Game.reset(number_uav, weapon_positions, uav_velocity, uav_position, mmap)

    @staticmethod
    def try_get_current_target():
        return Game.try_get_current_target()

    @staticmethod
    def get_uav_list():
        return Game.get_uav_list()

    @staticmethod
    def get_weapon_state():
        return Game.get_weapon_state()
