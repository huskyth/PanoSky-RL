from enum import Enum


class AbsEnum(Enum):
    @staticmethod
    def build_state(value, class_type):
        ret = [0 for x in range(len(class_type))]
        ret[value] = 1
        return ret


class UAVType(AbsEnum):
    DECOY = 0
    TASK = 1


class AttackState(AbsEnum):
    SAFE = 0
    ATTACKING = 1
    DESTROYED = 2


class UAVState(AbsEnum):
    ALIVE = 'ALIVE'
    DESTROYED = 'DESTROYED'
    COLLISION = 'COLLISION'
