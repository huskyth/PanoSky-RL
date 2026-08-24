import numpy as np

from onpolicy.utils.math_tool import normalize


class TrainUAV:
    def __init__(self, px, py, pz, vx, vy, vz, is_attacked_state, status):
        self.position = [px, py, pz]
        self.velocity = [vx, vy, vz]
        self.is_attacked_state = is_attacked_state
        self.status = status

    def get_position_and_velocity(self):
        return self.position + self.velocity

    def get_normalize_velocity(self):
        return normalize(self.velocity)

    def get_normalize_position(self, normal_):
        return (np.array(self.position) / normal_).tolist()

    def set_position(self, x, y, z):
        self.position = [x, y, z]

    def set_velocity(self, x, y, z):
        self.velocity = [x, y, z]

    def to_dict(self):
        return {
            'ID': id(self),
            "position": self.position,
            "velocity": self.velocity,
            "status": self.status.value,
            'is_attacked_state': str(self.is_attacked_state)
        }

    def __str__(self):
        return str({
            "position": self.position,
            "velocity": self.velocity,
            "status": self.status.value,
            'is_attacked_state': str(self.is_attacked_state)
        })
