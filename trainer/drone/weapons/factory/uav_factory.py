from trainer.drone.weapons.entries.uav.uav import UAV
from trainer.drone.weapons.factory.config_factory import ConfigFactory, ConfigEnum


class UAVFactory:
    @staticmethod
    def create():
        return UAV(ConfigFactory.create(config_type=ConfigEnum.uav))
