from onpolicy.envs.drone.weapons.entries.config.phalanx_components_config_factory import PhalanxComponentsConfigFactory, \
    PhalanxComponentConfigEnum
from onpolicy.utils.object_tool import *


class PhalanxConfig:
    def __init__(self, *keys, **kward):
        self.num = 0
        self.position = [1000, 0, 0]  # 密集阵坐标
        self.search_rader_config = PhalanxComponentsConfigFactory.create(PhalanxComponentConfigEnum.search_rader,
                                                                         position=self.position)
        self.track_rader_config = PhalanxComponentsConfigFactory.create(PhalanxComponentConfigEnum.track_rader,
                                                                        position=self.position)
        self.weapon_config = PhalanxComponentsConfigFactory.create(PhalanxComponentConfigEnum.weapon,
                                                                   position=self.position)

        if 'intercept_fun' in kward:
            assert 'config_name' in kward
            assert 'config_key' in kward
            assert 'config_value' in kward
            instance = ObjectTool.get_value_by_name(self, kward['config_name'])

            ObjectTool.set_value_by_name(instance, kward['config_key'], kward['config_value'])
