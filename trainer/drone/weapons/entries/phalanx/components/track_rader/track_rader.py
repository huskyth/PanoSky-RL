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

        # 武器基座位置
        self.position = config.common_config['position']
        temp = self.position.copy()
        temp[0] += 1
        self.top_project_position = temp  # 炮口指向位置（可视化）
        self.top_angle_with_xoy = 0

        self.rotate_angular_velocity = self.config.rotate_angular_velocity
        self.rotate_angular_acceleration = self.config.rotate_angular_acceleration

        # 调弦状态
        self.need_adjust_board_time = 0.0      # 剩余调弦时间
        self.initial_adjust_time = 0.0         # 调弦开始时总时间
        self.end_position = temp               # 调弦结束时炮口位置
        self.end_theta = 0

        # 目标相关
        self.current_target = None
        self.capture_time = 0
        self.search_rader = None

        # ===== 新增：调弦状态锁 =====
        self.is_tuning = False                 # True 表示调弦进行中

        # 调试记录
        self.last_target_position = None
        self.tuning_start_position = None

    def get_aim_point(self):
        """返回当前炮口指向的目标点（绝对坐标），用于可视化"""
        if hasattr(self, 'top_project_position'):
            return self.top_project_position
        else:
            return [self.position[0] + 1, self.position[1], self.position[2]]

    def _cal_time(self, theta):
        """计算调弦所需时间"""
        t1 = 2 * math.sqrt(theta / self.rotate_angular_acceleration)
        t2 = theta / self.rotate_angular_velocity + self.rotate_angular_velocity / self.rotate_angular_acceleration
        return min(t2, t1)

    def _compute_target_direction(self, target_pos):
        """计算从武器指向目标的三维方向向量（单位向量）"""
        if target_pos is None:
            return np.array([1.0, 0.0, 0.0])
        # 直接使用目标的三维位置
        uav_weapon_vector = subtraction_of_2_vector(target_pos, self.position)
        norm = np.linalg.norm(uav_weapon_vector)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        return np.array(uav_weapon_vector) / norm

    def _get_current_direction(self):
        """获取当前炮口指向方向（单位向量）"""
        current_direction = subtraction_of_2_vector(self.top_project_position, self.position)
        norm = np.linalg.norm(current_direction)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        return np.array(current_direction) / norm

    def _update_end_position(self, target_pos):
        """更新 end_position 为目标当前位置"""
        if target_pos is None:
            return
        target_dir = self._compute_target_direction(target_pos)
        self.end_position = add_of_2_vector(self.position, target_dir.tolist())

    def calculate_adjust_data(self, dynamic_target_position=None):
        """
        计算调弦数据
        - 如果调弦已结束，且没有正在进行的调弦，计算新的调弦并开始
        - 如果调弦正在进行，不重置计时器，只更新 end_position（指向目标当前位置）
        """
        if self.current_target is None and dynamic_target_position is None:
            logger.warning("调弦计算时，目标不存在")
            return

        # 1. 获取目标最新位置
        if dynamic_target_position is not None:
            target_pos = dynamic_target_position
        else:
            target_pos = self.current_target.position

        # 2. 计算当前和目标方向
        current_dir = self._get_current_direction()
        target_dir = self._compute_target_direction(target_pos)

        # 3. 计算水平方向需要转动的角度
        will_rotate_angle = cal_angle_of_2_vector(current_dir.tolist(), target_dir.tolist())

        # 4. 计算调弦时间
        horizontal_time = self._cal_time(will_rotate_angle)

        # 5. 垂直方向角度（俯仰）
        uav_projection_point = [target_pos[0], target_pos[1], self.position[2]]
        small = distance_of_2_point(self.position, uav_projection_point)
        large = distance_of_2_point(self.position, target_pos)
        if large > 0:
            uav_theta = radian_2_angle(math.acos(small / large))
        else:
            uav_theta = 0
        end_theta = abs(uav_theta) if target_pos[2] >= self.position[2] else -abs(uav_theta)
        vertical_time = self._cal_time(abs(self.top_angle_with_xoy - end_theta))

        adjust_time = max(horizontal_time, vertical_time)

        # ===== 核心逻辑：调弦状态机 =====
        if not self.is_tuning:
            # 调弦未开始，启动新的调弦
            self.need_adjust_board_time = adjust_time
            self.initial_adjust_time = adjust_time
            self.is_tuning = True
            self.tuning_start_position = self.top_project_position.copy()
            logger.info(f"🔵 调弦开始，目标方向，总时间 {adjust_time:.2f}s")
        else:
            # 调弦正在进行，不重置计时器！
            # 只更新 end_position（让调弦结束时指向目标当前位置）
            pass

        # ===== 每步更新 end_position（指向目标当前位置） =====
        self._update_end_position(target_pos)
        self.end_theta = end_theta

        # ===== 每步更新 top_project_position（可视化插值） =====
        if self.is_tuning and self.initial_adjust_time > 0:
            progress = 1.0 - self.need_adjust_board_time / self.initial_adjust_time
            progress = max(0.0, min(1.0, progress))

            current_dir_list = current_dir.tolist()
            target_dir_list = target_dir.tolist()
            interpolated_dir = [
                current_dir_list[0] + (target_dir_list[0] - current_dir_list[0]) * progress,
                current_dir_list[1] + (target_dir_list[1] - current_dir_list[1]) * progress,
                current_dir_list[2] + (target_dir_list[2] - current_dir_list[2]) * progress
            ]
            norm = np.linalg.norm(interpolated_dir)
            if norm > 0:
                interpolated_dir = [interpolated_dir[0]/norm, interpolated_dir[1]/norm, interpolated_dir[2]/norm]
            self.top_project_position = add_of_2_vector(self.position, interpolated_dir)
        else:
            # 调弦结束或异常情况，直接指向目标方向
            self.top_project_position = add_of_2_vector(self.position, target_dir.tolist())

        self.last_target_position = target_pos

        # logger.info(f"调弦计算：角度={will_rotate_angle:.2f}°, 时间={adjust_time:.2f}s, "
        #             f"剩余={self.need_adjust_board_time:.2f}s, 进度={progress:.2f}, is_tuning={self.is_tuning}")

    def adjust_board(self, fun):
        """
        调弦过程：每步减少计时器，直到调弦结束
        调弦一旦开始，不结束就不进行下一次调弦
        """
        if self.current_target is None:
            return True

        # 如果不在调弦状态，直接返回（没有调弦需要执行）
        if not self.is_tuning:
            return True

        # ===== 每步更新 end_position（指向目标当前位置） =====
        self._update_end_position(self.current_target.position)

        # ===== 每步更新可视化插值 =====
        self.calculate_adjust_data(self.current_target.position)

        # ===== 调弦时间减少（只减不增！） =====
        self._step_adjust_board_one_time()

        # ===== 判断调弦是否完成 =====
        if self.need_adjust_board_time <= 0:
            self.adjust_end_set_value()
            logger.info(f"✅ 调弦完成，炮口对准目标 {fun(self.current_target)}")
            return True

        logger.info(f"⏳ 调弦中，剩余时间 {self.need_adjust_board_time:.2f}s")
        return False

    def _step_adjust_board_one_time(self):
        """调弦步进，时间减少一个单位"""
        self.need_adjust_board_time -= UNIT_TIME
        if self.need_adjust_board_time < 0:
            self.need_adjust_board_time = 0

    def adjust_end_set_value(self):
        """调弦结束，更新炮口位置"""
        logger.info("🎯 调弦结束，炮口位置更新", is_in_file=False)

        # ===== end_position 已经在调弦过程中持续更新，指向目标当前位置 =====
        self.top_project_position = self.end_position
        self.top_angle_with_xoy = self.end_theta

        # 重置调弦状态
        self.need_adjust_board_time = 0.0
        self.initial_adjust_time = 0.0
        self.is_tuning = False
        self.tuning_start_position = None

        logger.info(f"  炮口指向位置: {self.top_project_position}")

    # ===== 以下方法保持不变 =====
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
        logger.info(f"移除目标，移除前子弹数 {n}, 移除后 {len(weapon.fired_bullet_list)}")
        self.current_target = None
        self.last_target_position = None
        self.tuning_start_position = None
        self.initial_adjust_time = 0.0
        self.is_tuning = False  # 目标移除时，调弦状态也重置

    def confirm_track_target(self, search_rader, uav_list, get_uav_index_fun):
        threat_level_uav_list = self._adjust_board_prepare(search_rader, uav_list, get_uav_index_fun)
        if 0 == len(threat_level_uav_list):
            logger.info("没有搜索到或者没有在跟踪范围内")
            return None

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

        # ===== 锁定目标时，如果不在调弦中，开始调弦 =====
        if self.current_target is not None and not self.is_tuning:
            self.tuning_start_position = self.top_project_position.copy()
            self.calculate_adjust_data(self.current_target.position)

        return self.current_target

    def _adjust_board_prepare(self, search_rader, uav_list, fun):
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