from onpolicy.envs.drone.weapons.entries.phalanx.phalanx import Phalanx
from onpolicy.envs.drone.weapons.factory.config_factory import ConfigFactory, ConfigEnum


class PhalanxFactory:
    phalanx_instance = None

    @staticmethod
    def create(*keys, **kward):
        if PhalanxFactory.phalanx_instance is not None:
            return PhalanxFactory.phalanx_instance
        PhalanxFactory.phalanx_instance = Phalanx(ConfigFactory.create(*keys, config_type=ConfigEnum.phalanx, **kward),
                                                  kward['map'])
        return PhalanxFactory.phalanx_instance

    @staticmethod
    def reset():
        PhalanxFactory.phalanx_instance = None
