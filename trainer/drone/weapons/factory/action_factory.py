from onpolicy.envs.drone.weapons.entries.actions import *


class ActionEnum(Enum):
    TimeStepAction = 0,
    SearchAction = 2,
    AdjustBoardAction = 3,
    CaptureAction = 4,
    LoadBulletAction = 5,
    FireAction = 6,
    BulletAttackAction = 7,
    ConfirmTargetAction = 8


class ActionFactory:
    @staticmethod
    def create(action_type):
        if action_type == ActionEnum.TimeStepAction:
            return TimeStepAction()
        elif action_type == ActionEnum.ConfirmTargetAction:
            return ConfirmTargetAction()
        elif action_type == ActionEnum.SearchAction:
            return SearchAction()
        elif action_type == ActionEnum.AdjustBoardAction:
            return AdjustBoardAction()
        elif action_type == ActionEnum.CaptureAction:
            return CaptureAction()
        elif action_type == ActionEnum.LoadBulletAction:
            return LoadBulletAction()
        elif action_type == ActionEnum.FireAction:
            return FireAction()
        elif action_type == ActionEnum.BulletAttackAction:
            return BulletAttackAction()
        else:
            raise Exception("not find Action type")
