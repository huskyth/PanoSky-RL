from onpolicy.envs.drone.weapons.entries.abstract_entry import AbstractEntry, logger
from onpolicy.envs.drone.weapons.entries.config.global_config import *
from onpolicy.utils.math_tool import *
from onpolicy.envs.drone.weapons.entries.uav.uav_enum import *


class UAV(AbstractEntry):
    def __init__(self, config):
        super().__init__(config)
        # 被搜索标记的次数
        self.is_attack = AttackState.SAFE
        self.search_marked_times = 0
        self.position = self.config.position
        self.horizontal_right_vector = [1, 0, 0]
        # velocity指速度大小
        self.velocity = self.config.velocity
        self.velocity_direction = normalize(self.config.velocity_direction)
        self.type = UAVType.TASK
        self.state = UAVState.ALIVE

    def set_uav_type(self, uav_type):
        self.type = uav_type

    def get_uav_type(self):
        return self.type

    def _cur_position_vector(self):
        result = self.position + [1]
        return result

    def set_attacked_state(self, attack_state):
        self.is_attack = attack_state

    def get_uav_state(self):
        return self.state

    def set_uav_state(self, state):
        self.state = state

    def get_attacked_state(self):
        return self.is_attack

    def reset_attacked_state(self):
        self.is_attack = AttackState.SAFE

    def _build_translate_matrix(self):
        result = [[0 for i in range(4)] for j in range(4)]
        distance_per_unit_time = scalar_mul_vector(self.velocity * UNIT_TIME, self.velocity_direction)
        for row in range(4):
            result[row][row] = 1
            if row <= 2:
                result[row][-1] = distance_per_unit_time[row]
        return result

    def uav_fly_a_unit_time(self):
        translate_matrix = self._build_translate_matrix()
        cur_position_vector = self._cur_position_vector()
        self.position = matrix_mul_vector(translate_matrix, cur_position_vector)[:-1]

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position.copy()

    def set_velocity(self, velocity):
        self.velocity = length_of_vector(velocity)
        self.velocity_direction = normalize(velocity.copy())

    def print_self(self):
        return 'id为：' + self.get_id() + ", 被标记次数为：" + str(self.search_marked_times) + "  "

    def get_velocity_size(self):
        return self.velocity

    def get_velocity(self):
        '''
        蕴含大小信息
        '''
        from onpolicy.utils.math_tool import scalar_mul_vector
        return scalar_mul_vector(self.velocity, self.velocity_direction)
