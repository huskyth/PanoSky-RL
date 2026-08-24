from onpolicy.envs.drone.weapons.entries.abstract_entry import *
from onpolicy.utils.random_event import *

from onpolicy.envs.drone.weapons.entries.config.global_config import *
from onpolicy.envs.drone.weapons.entries.uav.uav_enum import *
import os
from onpolicy.utils.format_logger import AppLogger
from onpolicy.utils.util import compute_distance

logger = AppLogger().get_logger()


class Weapon(AbstractEntry):
    def __init__(self, config):
        super().__init__(config)
        self.time_instruction = 0
        self.current_bullet_num = config.BULLET_CAPACITY
        self.position = config.position
        self.bullet_velocity = config.bullet_velocity
        self.hit_kill_probability = config.hit_kill_probability
        self.bullet_fire_speed = config.bullet_fire_speed
        self.bullet_load_speed = config.bullet_load_speed
        self.fired_bullet_list = []
        self.is_fire_instruction_finish_bool = False
        self.bullet_load_time = config.BULLET_CAPACITY / config.bullet_load_speed / UNIT_TIME
        self.current_bullet_load_time = 0

    def build_dict(self):
        return self.config.build_dict()

    def clear_fired_bullet_list(self):
        self.fired_bullet_list = []

    class BulletState(Enum):
        NO_KILLED_NO_USE = 0
        FLYING_USEING = 1
        KILLED_NO_USE = 2

    def _fire_some_bullet(self, number, target):
        for n in range(number):
            self.fired_bullet_list.append(self._build_a_bullet(target))

    def _build_a_bullet(self, target):
        return Bullet(None, target, self.position, self.bullet_velocity, self.hit_kill_probability,
                      self.fired_bullet_list)

    def fire(self, target):
        fire_number = ceil_number(UNIT_TIME * self.bullet_fire_speed)
        fire_number = int(min(fire_number, self.current_bullet_num))
        self._fire_some_bullet(fire_number, target)
        self.current_bullet_num -= fire_number
        logger.info(
            str(os.getpid()) + "发射了" + str(fire_number) + "枚子弹，还剩下" + str(self.current_bullet_num) + "枚子弹")

    def step_time_instruction(self):
        self.time_instruction += UNIT_TIME
        logger.info("指令准备时间自增，当前指令时间：" + str(self.time_instruction))

    def reset_time_instruction(self):
        self.time_instruction = 0

    def is_fire_instruction_finish(self):
        if self.is_fire_instruction_finish_bool: return True
        _is_fire_instruction_finish = self.time_instruction == FIRE_INSTRUCTION_TIME
        if _is_fire_instruction_finish:
            logger.info("指令准备完成，当前指令时间为：" + str(self.time_instruction))
            self.is_fire_instruction_finish_bool = True
        return _is_fire_instruction_finish

    def reset_fire_instruction(self):
        self.reset_time_instruction()
        self.is_fire_instruction_finish_bool = False

    def is_bullet_can_fire(self):
        logger.info(str(os.getpid()) + "判断子弹有多少的时候的弹量为{}".format(self.current_bullet_num),
                    is_in_file=False)
        return self.current_bullet_num > 0

    def step_load_bullet(self):
        '''
        装弹一次
        :return: 返回装弹是否完成
        '''
        self.current_bullet_load_time += UNIT_TIME
        if self.current_bullet_load_time >= self.bullet_load_time:
            logger.info("装弹完成")
            self.current_bullet_num = self.config.BULLET_CAPACITY
            self.current_bullet_load_time = 0
            return True
        else:
            logger.info("还在装弹，当前装弹时间为{}，还剩下的装弹时间为{}，总共花的装弹时间为{}".format(
                self.current_bullet_load_time, self.bullet_load_time - self.current_bullet_load_time,
                self.bullet_load_time), is_in_file=False)
            return False


# class AttackState:
#     pass


class Bullet(AbstractEntry):
    '''
    子弹类被武器类持有
    '''

    def __init__(self, config, target, position, velocity, hit_kill_probability, fired_bullet_list):
        '''

        :param config:
        :param target:
        :param position: 子弹当前位置，就是雷达的位置
        :param velocity:
        :param hit_probability:
        :param kill_probability:
        :param remove_function: 同时移除无人机和子弹
        '''
        super().__init__(config)
        self.target = target
        self.position = position
        self.velocity = velocity
        self.hit_kill_probability = hit_kill_probability
        self.fired_bullet_list = fired_bullet_list
        self.all_time_to_fly = self._calculate_all_time_of_fly()
        self.target.set_attacked_state(AttackState.ATTACKING)
        logger.info(f"初始化的时候飞机位置 {target.position}, 当前和武器距离{compute_distance(self.position, self.target.position)}")

    def _calculate_all_time_of_fly(self):
        distance = distance_of_2_point(self.target.position, self.position)
        return distance / self.velocity

    def step_attack_a_target_and_is_kill(self, uav_list, fun):
        logger.info("id为：" + self.get_id() + "的子弹还需要飞行的时间：" + str(self.all_time_to_fly))

        if self.all_time_to_fly <= UNIT_TIME:
            # 时间到了，进行毁伤计算
            is_hit_and_kill = self.is_hit_kill_by_mc()
            logger.info(
                "id为：" + self.get_id() + "的子弹" + "它的无人机为：" + self.target.get_id() + ", 参数为" + str(
                    self.hit_kill_probability) + "，蒙特卡洛是否被 命中和毁伤 ：" + str(
                    is_hit_and_kill) + f'当前和武器距离{compute_distance(self.position, self.target.position)}')
            # 命中
            if is_hit_and_kill:
                # 被毁伤，移除自己
                self.target.set_attacked_state(AttackState.DESTROYED)
                logger.info(
                    "id为" + fun(self.target) + "的无人机被摧毁,当前随机数n为" + str(GameConfig.RANDOM_N),
                    f"当前无人机列表ID {[id(x) for x in uav_list]}",
                    is_in_file=False)
                self.target.remove_self_from_list(uav_list)

                logger.info(f"移除后无人机列表ID {[id(x) for x in uav_list]}， {uav_list}")
                return Weapon.BulletState.KILLED_NO_USE
            else:
                self.target.set_attacked_state(AttackState.SAFE)
                # 子弹失效
                return Weapon.BulletState.NO_KILLED_NO_USE
        else:
            logger.info("id为：" + self.get_id() + "的子弹飞行中")
            # 正在飞行中
            self.all_time_to_fly -= UNIT_TIME
            return Weapon.BulletState.FLYING_USEING

    def is_hit_kill_by_mc(self):
        is_hit = single_probability_event(self.hit_kill_probability)
        return is_hit
