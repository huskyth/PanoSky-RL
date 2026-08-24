from onpolicy.envs.drone.weapons.entries.uav.uav import UAV
from onpolicy.envs.drone.weapons.factory.config_factory import ConfigFactory, ConfigEnum


class UAVFactory:
    @staticmethod
    def create():
        return UAV(ConfigFactory.create(config_type=ConfigEnum.uav))
