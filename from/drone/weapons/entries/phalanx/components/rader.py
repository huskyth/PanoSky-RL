from onpolicy.envs.drone.weapons.entries.abstract_entry import AbstractEntry, logger
from onpolicy.envs.drone.weapons.entries.config.global_config import *
from onpolicy.utils.math_tool import *


class Rader(AbstractEntry):
    def __init__(self, config):
        super().__init__(config)
