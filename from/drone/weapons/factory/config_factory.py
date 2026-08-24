from onpolicy.envs.drone.weapons.entries.config.phalanx_config import *
from onpolicy.envs.drone.weapons.entries.config.uav_config import UAVConfig
from enum import Enum


class ConfigEnum(Enum):
    phalanx = 1
    uav = 2


class ConfigFactory:
    phalanxConfig = None
    uavConfig = None

    @staticmethod
    def create(*keys, **kward):
        assert 'config_type' in kward, 'no config_type'
        config_type = kward['config_type']
        if config_type == ConfigEnum.phalanx:
            if ConfigFactory.phalanxConfig is None:
                ConfigFactory.phalanxConfig = PhalanxConfig(*keys, **kward)
            return ConfigFactory.phalanxConfig
        elif config_type == ConfigEnum.uav:
            if ConfigFactory.uavConfig is None:
                ConfigFactory.uavConfig = UAVConfig()
            return ConfigFactory.uavConfig
        else:
            raise Exception("not find config type")

    @staticmethod
    def reset():
        ConfigFactory.phalanxConfig = None
        ConfigFactory.uavConfig = None


if __name__ == '__main__':
    pass
