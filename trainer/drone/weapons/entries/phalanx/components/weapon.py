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
    子弹类 - 支持爆炸半径范围伤害
    '''

    def __init__(self, target, weapon_position, bullet_velocity, hit_kill_probability,
                 fired_bullet_list, explosion_radius=50.0):
        '''
        :param target: 原始锁定目标
        :param weapon_position: 武器发射位置
        :param bullet_velocity: 子弹速度
        :param hit_kill_probability: 单发毁伤概率（当爆炸半径外或单目标模式时使用）
        :param fired_bullet_list: 子弹列表引用
        :param explosion_radius: 爆炸半径（米），<=0 表示无爆炸半径，使用单目标概率判定
        '''
        super().__init__(None)
        self.target = target
        self.position = np.array(weapon_position, dtype=float)
        self.velocity = bullet_velocity
        self.hit_kill_probability = hit_kill_probability
        self.fired_bullet_list = fired_bullet_list

        # ===== 新增：爆炸半径 =====
        self.explosion_radius = explosion_radius

        # 计算飞行时间（到锁定目标的距离 / 速度）
        self.all_time_to_fly = self._calculate_all_time_of_fly()

        # 锁定目标进入“被攻击”状态
        if target is not None:
            target.set_attacked_state(AttackState.ATTACKING)
            logger.info(
                f"初始化的时候飞机位置 {target.position}, 当前和武器距离{compute_distance(self.position, self.target.position)}")
        else:
            logger.warning("子弹创建时 target 为 None")

    def _calculate_all_time_of_fly(self):
        if self.target is None:
            return float('inf')
        distance = distance_of_2_point(self.target.position, self.position)
        return distance / self.velocity

    # ============================================================
    # 核心改动：step_attack_a_target_and_is_kill（支持爆炸半径）
    # ============================================================
    def step_attack_a_target_and_is_kill(self, uav_list, fun):
        """
        子弹飞行每步调用，检查是否到达目标位置。
        如果到达：
          1. 如果有爆炸半径 -> 对爆炸半径内的所有无人机进行毁伤判定
          2. 如果无爆炸半径 -> 只对锁定目标进行概率判定
        :param uav_list: 所有存活的无人机列表（用于范围伤害）
        :param fun: 获取无人机ID的函数
        :return: BulletState
        """
        if self.target is None:
            logger.warning("子弹目标已丢失，子弹失效")
            return Weapon.BulletState.NO_KILLED_NO_USE

        logger.info(f"id为：{self.get_id()} 的子弹还需要飞行的时间：{self.all_time_to_fly}")

        # ---- 1. 如果子弹还在飞行中 ----
        if self.all_time_to_fly > UNIT_TIME:
            self.all_time_to_fly -= UNIT_TIME
            logger.info(f"id为：{self.get_id()} 的子弹飞行中")
            return Weapon.BulletState.FLYING_USEING

        # ---- 2. 子弹到达目标位置 ----
        # 计算子弹落点（使用目标当前位置，或目标发射时的位置）
        impact_point = np.array(self.target.position, dtype=float)

        # ============================================================
        # 爆炸半径模式：对范围内的所有无人机进行判定
        # ============================================================
        if self.explosion_radius > 0:
            logger.info(f"id为：{self.get_id()} 的子弹到达目标 {impact_point}，爆炸半径 {self.explosion_radius}m")
            hit_any = False

            # 遍历所有无人机（复制列表，避免在遍历中修改）
            for uav in uav_list[:]:
                if uav is None:
                    continue
                dist_to_impact = compute_distance(uav.position, impact_point)

                # 如果无人机在爆炸半径内
                if dist_to_impact <= self.explosion_radius:
                    # 进行毁伤判定
                    is_killed = single_probability_event(self.hit_kill_probability)
                    logger.info(
                        f"无人机 {fun(uav)} 在爆炸半径内 (距离={dist_to_impact:.1f}m)，"
                        f"命中概率 {self.hit_kill_probability}，判定结果：{'击毁' if is_killed else '未击毁'}"
                    )

                    if is_killed:
                        uav.set_attacked_state(AttackState.DESTROYED)
                        logger.info(f"id为 {fun(uav)} 的无人机被爆炸摧毁")
                        uav.remove_self_from_list(uav_list)
                        hit_any = True
                else:
                    logger.info(f"无人机 {fun(uav)} 在爆炸半径外 (距离={dist_to_impact:.1f}m)，安全")

            # 如果命中了任何无人机，子弹失效
            if hit_any:
                # 重置所有未被摧毁的无人机的攻击状态
                for uav in uav_list:
                    if uav is not None and uav.attacked_state != AttackState.DESTROYED:
                        uav.set_attacked_state(AttackState.SAFE)
                return Weapon.BulletState.KILLED_NO_USE
            else:
                # 没有命中任何无人机
                for uav in uav_list:
                    if uav is not None:
                        uav.set_attacked_state(AttackState.SAFE)
                logger.info(f"爆炸半径内未命中任何目标，子弹失效")
                return Weapon.BulletState.NO_KILLED_NO_USE

        # ============================================================
        # 原有单目标模式：只对锁定目标进行判定
        # ============================================================
        else:
            is_hit_and_kill = single_probability_event(self.hit_kill_probability)
            logger.info(
                f"id为：{self.get_id()} 的子弹到达，无人机：{self.target.get_id()}，"
                f"命中概率 {self.hit_kill_probability}，判定结果：{is_hit_and_kill}"
            )

            if is_hit_and_kill:
                self.target.set_attacked_state(AttackState.DESTROYED)
                logger.info(f"id为 {fun(self.target)} 的无人机被摧毁")
                self.target.remove_self_from_list(uav_list)
                return Weapon.BulletState.KILLED_NO_USE
            else:
                self.target.set_attacked_state(AttackState.SAFE)
                return Weapon.BulletState.NO_KILLED_NO_USE

    def is_hit_kill_by_mc(self):
        """保留原有方法，用于兼容"""
        return single_probability_event(self.hit_kill_probability)