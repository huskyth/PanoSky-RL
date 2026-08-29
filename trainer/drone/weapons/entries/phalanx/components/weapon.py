import copy

import numpy as np

from trainer.drone.weapons.entries.abstract_entry import *
from trainer.utils.random_event import *
from trainer.drone.weapons.entries.config.global_config import *
from trainer.drone.weapons.entries.uav.uav_enum import *
import os
from trainer.utils.format_logger import AppLogger
from trainer.utils.util import compute_distance

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

        # ===== 新增：爆炸半径配置（可在config里设置） =====
        self.explosion_radius = getattr(config, 'explosion_radius', 50.0)  # 默认50米

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
        return Bullet(
            target=target,
            weapon_position=self.position,
            bullet_velocity=self.bullet_velocity,
            hit_kill_probability=self.hit_kill_probability,
            fired_bullet_list=self.fired_bullet_list,
            explosion_radius=self.explosion_radius  # ===== 传递爆炸半径 =====
        )

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
        if self.is_fire_instruction_finish_bool:
            return True
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


class Bullet(AbstractEntry):
    '''
    子弹类 - 支持爆炸半径范围伤害，实时更新飞行位置
    '''

    def __init__(self, target, weapon_position, bullet_velocity, hit_kill_probability,
                 fired_bullet_list, explosion_radius=50.0):
        super().__init__(None)
        self.target = target
        self.weapon_position = np.array(weapon_position, dtype=float)
        self.current_position = np.array(weapon_position, dtype=float)  # 实时位置
        self.velocity = bullet_velocity
        self.hit_kill_probability = hit_kill_probability
        self.fired_bullet_list = fired_bullet_list
        self.explosion_radius = explosion_radius

        # 计算总飞行时间
        self.all_time_to_fly = self._calculate_all_time_of_fly()
        self.elapsed_time = 0.0  # 已飞行时间

        # 记录目标发射时的位置（作为参考）
        self.impact_point = np.array(copy.deepcopy(self.target.position), dtype=float)

        if target is not None:
            target.set_attacked_state(AttackState.ATTACKING)
            logger.info(f"子弹创建，目标位置: {target.position}, 武器位置: {weapon_position}")

    def _calculate_all_time_of_fly(self):
        if self.target is None:
            return float('inf')
        self.weapon_position = self.weapon_position.tolist() if isinstance(self.weapon_position,
                                                                           np.ndarray) else self.weapon_position
        distance = distance_of_2_point(self.target.position, self.weapon_position)
        return distance / self.velocity

    def step_attack_a_target_and_is_kill(self, uav_list, fun):
        """
        每步调用，更新子弹位置并检查是否命中
        """
        if self.target is None:
            logger.warning("子弹目标已丢失，子弹失效")
            return Weapon.BulletState.NO_KILLED_NO_USE

        # ---- 1. 更新子弹实时位置 ----
        self.elapsed_time += UNIT_TIME

        # 如果子弹还未到达目标，计算插值位置
        if self.all_time_to_fly > 0 and self.elapsed_time < self.all_time_to_fly:
            progress = self.elapsed_time / self.all_time_to_fly
            # 从武器位置到目标位置的插值
            self.current_position = self.weapon_position + (self.impact_point - self.weapon_position) * progress
        else:
            # 超过飞行时间，子弹到达目标位置
            self.current_position = self.impact_point.copy()
            self.all_time_to_fly = 0  # 标记已到达

        # ---- 2. 判断是否到达或超过目标 ----
        if self.elapsed_time >= self.all_time_to_fly and self.all_time_to_fly > 0:
            self.all_time_to_fly = 0  # 到达

        # ---- 3. 如果子弹还在飞行中，只更新位置，不判定爆炸 ----
        if self.elapsed_time < self._calculate_all_time_of_fly():
            return Weapon.BulletState.FLYING_USEING

        # ---- 4. 子弹到达目标位置，进行爆炸判定 ----
        logger.info(f"子弹到达目标位置: {self.current_position}")

        if self.explosion_radius > 0:
            hit_any = False
            for uav in uav_list[:]:
                if uav is None:
                    continue
                dist_to_impact = compute_distance(uav.position, self.current_position)

                if dist_to_impact <= self.explosion_radius:
                    is_killed = single_probability_event(self.hit_kill_probability)

                    if is_killed:
                        uav.set_attacked_state(AttackState.DESTROYED)
                        logger.info(
                            f"无人机 {fun(uav)} 在爆炸半径内 (距离={dist_to_impact:.1f}m)，"
                            f"命中概率 {self.hit_kill_probability}，判定结果：{'击毁' if is_killed else '未击毁'}"
                            f"{self.target}---------{uav}"
                        )
                        uav.remove_self_from_list(uav_list)
                        hit_any = True
                else:
                    logger.info(f"无人机 {fun(uav)} 在爆炸半径外 (距离={dist_to_impact:.1f}m)，安全")

            if hit_any:
                for uav in uav_list:
                    if uav is not None and uav.get_attacked_state() != AttackState.DESTROYED:
                        uav.set_attacked_state(AttackState.SAFE)
                return Weapon.BulletState.KILLED_NO_USE
            else:
                for uav in uav_list:
                    if uav is not None:
                        uav.set_attacked_state(AttackState.SAFE)
                logger.info("爆炸半径内未命中任何目标")
                return Weapon.BulletState.NO_KILLED_NO_USE

        else:
            # 单目标模式
            is_hit_and_kill = single_probability_event(self.hit_kill_probability)
            if is_hit_and_kill:
                self.target.set_attacked_state(AttackState.DESTROYED)
                self.target.remove_self_from_list(uav_list)
                return Weapon.BulletState.KILLED_NO_USE
            else:
                self.target.set_attacked_state(AttackState.SAFE)
                return Weapon.BulletState.NO_KILLED_NO_USE
