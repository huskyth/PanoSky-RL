from onpolicy.envs.drone.weapons.entries.phalanx.components.search_rader import *
from onpolicy.envs.drone.weapons.entries.phalanx.components.track_rader.track_rader import *
from onpolicy.envs.drone.weapons.entries.phalanx.components.weapon import *
from onpolicy.envs.drone.weapons.entries.abstract_entry import AbstractEntry


class Phalanx(AbstractEntry):
    def __init__(self, config, mmap):
        super().__init__(config)
        self.search_rader, self.track_rader, self.weapon = SearchRader(config.search_rader_config, mmap), TrackRader(
            config.track_rader_config, mmap), Weapon(config.weapon_config)
        self.track_rader.set_search_rader(self.search_rader)

    def set_position(self, position):
        self.search_rader.position = position.copy()
        self.track_rader.position = position.copy()
        self.weapon.position = position.copy()

    def get_position(self):
        return self.search_rader.position.copy()
