from onpolicy.utils.format_logger import AppLogger

logger = AppLogger().get_logger()

from onpolicy.envs.drone.weapons.entries.config.components_configs import *

from enum import Enum


class PhalanxComponentConfigEnum(Enum):
    search_rader = 1
    track_rader = 2
    weapon = 3


class PhalanxComponentsConfigFactory:
    @staticmethod
    def create(config_type, **common_config):
        if config_type == PhalanxComponentConfigEnum.search_rader:
            ret_config = SearchRaderConfig(**common_config)
        elif config_type == PhalanxComponentConfigEnum.track_rader:
            ret_config = TrackRaderConfig(**common_config)
        elif config_type == PhalanxComponentConfigEnum.weapon:
            ret_config = WeaponConfig(**common_config)
        else:
            raise Exception("not find config type")
        return ret_config
