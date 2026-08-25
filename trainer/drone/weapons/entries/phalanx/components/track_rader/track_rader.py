import numpy as np

from trainer.drone.weapons.entries.phalanx.components.rader import *
import math


class TrackRader(Rader):
    def __init__(self, config, mmap):
        super().__init__(config)
        self.map = mmap
        self.track_distance = self.config.track_distance
        self.fire_distance = self.config.fire_distance
        self.minimum_fire_distance = self.config.minimum_fire_distance
        self.minimum_track_distance = self.config.minimum_track_distance
        GameConfig.THREAT_LEVEL_THRESHOLD = 0

        # 舷基座的位置，相对于绝对坐标
        self.position = config.common_config['position']
        temp = self.position.copy()
        temp[0] += 1
        self.top_project_position = temp
        self.top_angle_with_xoy = 0
        # self.top_angle_with_xoy 为角度，且为与水平面夹角，上为正方向
        self.rotate_angular_velocity = self.config.rotate_angular_velocity
        self.rotate_angular_acceleration = self.config.rotate_angular_acceleration
        self.need_adjust_board_time = 0
        self.end_theta = 0
        self.end_position = 0

        self.current_target = None
        self.capture_time = 0
        self.search_rader = None

        # ===== 新增：记录上一次的目标位置，用于计算调弦是否追上 =====
        self.last_target_position = None
        self.tuning_start_position = None  # 调弦开始时炮口位置

    def _cal_time(self, theta):
        """
        计算调弦所需时间
        :param theta: 需要转动的角度（度）
        """
        t1 = 2 * math.sqrt(theta / self.rotate_angular_acceleration)
        t2 = theta / self.rotate_angular_velocity + self.rotate_angular_velocity / self.rotate_angular_acceleration
        return min(t2, t1)

    # ================================================================
    # 核心改动：calculate_adjust_data 支持持续追踪
    # ================================================================
    def calculate_adjust_data(self, dynamic_target_position=None):
        """
        计算调弦数据——每次调用时使用目标的最新位置
        :param dynamic_target_position: 目标当前位置（如果为None则使用current_target.position）
        """
        if self.current_target is None and dynamic_target_position is None:
            logger.warning("调弦计算时，目标不存在")
            return

        # 1. 获取目标最新位置
        if dynamic_target_position is not None:
            target_pos = dynamic_target_position
        else:
            target_pos = self.current_target.position

        # 2. 计算目标在武器高度上的水平投影点
        uav_projection_point = [target_pos[0], target_pos[1], self.position[2]]
        uav_weapon_vector = subtraction_of_2_vector(uav_projection_point, self.position)

        # 3. 当前炮口指向方向
        current_direction = subtraction_of_2_vector(self.top_project_position, self.position)
        current_direction_norm = np.linalg.norm(current_direction)
        if current_direction_norm < 1e-8:
            current_direction = np.array([1.0, 0.0, 0.0])  # 默认指向X轴
        else:
            current_direction = current_direction / current_direction_norm

        # 4. 目标方向（单位向量）
        target_direction = uav_weapon_vector / (np.linalg.norm(uav_weapon_vector) + 1e-8)

        # 5. 计算水平方向需要转动的角度
        will_rotate_angle = cal_angle_of_2_vector(current_direction.tolist(), target_direction.tolist())

        # 6. 计算调弦时间
        horizontal_time = self._cal_time(will_rotate_angle)

        # 7. 垂直方向角度（俯仰）
        small = distance_of_2_point(self.position, uav_projection_point)
        large = distance_of_2_point(self.position, target_pos)
        if large > 0:
            uav_theta = radian_2_angle(math.acos(small / large))
        else:
            uav_theta = 0
        end_theta = abs(uav_theta) if target_pos[2] >= self.position[2] else -abs(uav_theta)
        vertical_time = self._cal_time(abs(self.top_angle_with_xoy - end_theta))

        # 8. 取水平/垂直的最大时间
        adjust_time = max(horizontal_time, vertical_time)

        # 9. 更新目标位置（调弦结束后炮口应该指向的位置）
        self.end_position = add_of_2_vector(self.position, target_direction.tolist())
        self.end_theta = end_theta

        # 10. 更新调弦计时器（如果当前调弦时间比新计算的时间短，则延长）
        #    这是关键——当目标移动时，调弦时间会动态延长！
        if adjust_time > self.need_adjust_board_time:
            self.need_adjust_board_time = adjust_time

        # 记录目标位置，用于判断目标是否在移动
        self.last_target_position = target_pos

        logger.info(f"调弦计算：水平角={will_rotate_angle:.2f}°, 时间={horizontal_time:.2f}s, "
                    f"俯仰角={end_theta:.2f}°, 总调弦时间={adjust_time:.2f}s")

    def adjust_end_set_value(self):
        """调弦结束，更新炮口位置"""
        logger.info("调弦结束，更新炮口位置", is_in_file=False)
        self.top_project_position = self.end_position
        self.top_angle_with_xoy = self.end_theta
        self.need_adjust_board_time = 0
        self.tuning_start_position = None

    def set_search_rader(self, search_rader):
        self.search_rader = search_rader

    def get_current_target(self):
        assert self.current_target is not None, "current target is None"
        return self.current_target

    def try_get_current_target(self):
        return self.current_target

    def check_can_fire(self):
        can_fire = (self.capture_time == CAPTURE_TIME)
        if can_fire:
            logger.info("可以进入开火状态，捕获时间为：" + str(self.capture_time))
            self.capture_time = 0
        return can_fire

    def step_capture(self):
        self.capture_time += 1
        logger.info("捕获时间自增，当前捕获时间是：" + str(self.capture_time))

    def reset_capture_time(self):
        self.capture_time = 0

    def current_target_exist(self):
        return self.current_target is not None

    def remove_target(self, weapon):
        """移除目标，清空相关子弹"""
        if self.current_target is None:
            logger.info("移除目标时，当前目标不存在")
            return
        self.current_target.reset_attacked_state()
        contain_list = []
        n = len(weapon.fired_bullet_list)
        for bullet in weapon.fired_bullet_list:
            if bullet.target is not self.current_target:
                contain_list.append(bullet)
        weapon.fired_bullet_list = contain_list
        logger.info(
            f"移除目标，移除前子弹数 {n}, 移除后 {len(weapon.fired_bullet_list)}")
        self.current_target = None
        self.last_target_position = None
        self.tuning_start_position = None

    def confirm_track_target(self, search_rader, uav_list, get_uav_index_fun):
        """确认跟踪目标"""
        threat_level_uav_list = self._adjust_board_prepare(search_rader, uav_list, get_uav_index_fun)
        if 0 == len(threat_level_uav_list):
            logger.info("没有搜索到或者没有在跟踪范围内")
            return None
        else:
            logger.info("搜索到或者在跟踪范围内")

        max_threat_level = threat_level_uav_list[0][0]
        self.current_target = threat_level_uav_list[0][1]
        threat_level_string = ["威胁程度：" + str(entry[0]) + ", 对应无人机id：" + get_uav_index_fun(entry[1]) + "; " for
                               entry in threat_level_uav_list]
        logger.info("威胁程度概述：" + str(threat_level_string), is_in_file=False)

        for entry in threat_level_uav_list:
            is_in = self.search_rader.is_a_uav_in_search_range(entry[1])
            if max_threat_level < entry[0] and is_in:
                max_threat_level = entry[0]
                self.current_target = entry[1]

        if max_threat_level >= GameConfig.THREAT_LEVEL_THRESHOLD:
            logger.info("找到了最大威胁程度大于阈值 " + str(GameConfig.THREAT_LEVEL_THRESHOLD) +
                        "的无人机：" + self.current_target.print_self())
            assert self.current_target is not None, "_handle_equal_condition function exception"
        else:
            logger.info("没有找到，因为最大威胁程度小于阈值 " + str(GameConfig.THREAT_LEVEL_THRESHOLD))
            self.current_target = None

        # ===== 锁定目标时，立即计算调弦数据 =====
        if self.current_target is not None:
            self.tuning_start_position = self.top_project_position.copy()
            self.calculate_adjust_data(self.current_target.position)

        return self.current_target

    # ================================================================
    # 核心改动：adjust_board 每步都用目标最新位置重新计算
    # ================================================================
    def adjust_board(self, fun):
        """
        调弦过程：每步更新目标位置，直到炮口追上
        :param fun: 获取无人机ID的函数（用于日志）
        :return: True 表示调弦完成
        """
        if self.current_target is None:
            return True

        # ===== 关键：每步用目标当前位置重新计算调弦 =====
        self.calculate_adjust_data(self.current_target.position)

        # 调弦时间减少
        self._step_adjust_board_one_time()

        end = self.need_adjust_board_time <= 0
        if end:
            self.adjust_end_set_value()
            logger.info(f"调弦完成，炮口对准目标 {fun(self.current_target)}")
            return True

        logger.info(f"调弦中，剩余时间 {self.need_adjust_board_time:.2f}s")
        return False

    def _step_adjust_board_one_time(self):
        """调弦步进，时间减少一个单位"""
        self.need_adjust_board_time -= UNIT_TIME

    def _adjust_board_prepare(self, search_rader, uav_list, fun):
        """
        准备调弦数据：获取所有在跟踪范围内且标记>=3次的无人机
        :return: 返回元组列表，元组内容为（威胁程度，对应无人机）
        """
        uav_to_track_list = search_rader.get_mark_3_time_uav_list(uav_list)
        uav_to_track_list = [a_uav for a_uav in uav_to_track_list if self._is_a_uav_in_track_range(a_uav, fun)]
        threat_level_uav_list = [(self._cal_threat_level(a_uav), a_uav) for a_uav in uav_to_track_list]
        return threat_level_uav_list

    def _is_a_uav_in_track_range(self, a_uav, fun):
        logger.info(
            "最小跟踪距离为：" + str(self.minimum_track_distance) + "，跟踪最远距离为：" + str(
                self.track_distance) + "，与id为" + fun(a_uav) + "的无人机的距离：" + str(
                distance_of_2_point(self.position, a_uav.position)), is_in_file=False)
        return is_a_point_in_a_sphere(self.track_distance, self.position, a_uav.position) and is_a_point_out_a_sphere(
            self.minimum_track_distance, self.position, a_uav.position)

    def is_target_in_track_range(self, fun):
        assert self.current_target is not None, "current target is None"
        logger.info(
            "最小跟踪距离为：" + str(self.minimum_track_distance) + "，跟踪最远距离为：" + str(
                self.track_distance) + "，与id为" + fun(self.current_target) + "的无人机的距离：" + str(
                distance_of_2_point(self.position, self.current_target.position)), is_in_file=False)
        return is_a_point_in_a_sphere(self.track_distance, self.position,
                                      self.current_target.position) and is_a_point_out_a_sphere(
            self.minimum_track_distance, self.position, self.current_target.position)

    def is_target_can_use_because_no_mountain(self):
        assert self.current_target is not None, "current target is None"
        is_in = self.search_rader.is_a_uav_in_search_range(self.current_target)
        return is_in

    def is_target_in_fire_range(self):
        assert self.current_target is not None, "current target is None"
        logger.info("最小开火距离为：" + str(self.minimum_fire_distance) + "，开火最远距离为：" + str(
            self.fire_distance) + "，与id为" + self.current_target.get_id() + "无人机的距离：" + str(
            distance_of_2_point(self.position, self.current_target.position)))
        return is_a_point_in_a_sphere(self.fire_distance, self.position,
                                      self.current_target.position) and is_a_point_out_a_sphere(
            self.minimum_fire_distance, self.position, self.current_target.position)

    def _cal_threat_level(self, single_uav):
        return cal_threat_level(self.position, single_uav.velocity, single_uav.velocity_direction, single_uav.position)

    def _cal_projection_velocity(self, single_uav):
        return cal_projection_velocity(single_uav, self.position)