from onpolicy.envs.drone.weapons.entries.phalanx.components.rader import *
from onpolicy.utils.util import compute_distance


class TrackRader(Rader):
    def __init__(self, config, mmap):
        super().__init__(config)
        self.map = mmap
        self.track_distance = self.config.track_distance
        self.fire_distance = self.config.fire_distance
        self.minimum_fire_distance = self.config.minimum_fire_distance
        self.minimum_track_distance = self.config.minimum_track_distance
        # TODO://速度需要修改，这个5是速度常量
        GameConfig.THREAT_LEVEL_THRESHOLD = 0

        # 舷基座的位置，相对于绝对坐标
        self.position = config.common_config['position']
        temp = self.position.copy()
        temp[0] += 1
        self.top_project_position = temp
        self.top_angle_with_xoy = 0
        # self.top_angle_with_xoy 为角度，且为与水平面夹脚，上为正方向
        self.rotate_angular_velocity = self.config.rotate_angular_velocity
        self.rotate_angular_acceleration = self.config.rotate_angular_acceleration
        self.need_adjust_board_time = 0
        self.end_theta = 0
        self.end_position = 0

        self.current_target = None
        # 度/秒，单位度
        # 舷头此时的位置，相对于绝对坐标系
        self.capture_time = 0
        self.search_rader = None

    def _cal_time(self, theta):
        '''
        Todo: theta为度
        '''
        t1 = 2 * sqrt(theta / self.rotate_angular_acceleration)
        t2 = theta / self.rotate_angular_velocity + self.rotate_angular_velocity / self.rotate_angular_acceleration
        return min(t2, t1)

    def adjust_end_set_value(self):
        logger.info("调弦结束，赋值", is_in_file=False)
        self.top_project_position = self.end_position
        self.top_angle_with_xoy = self.end_theta
        self.need_adjust_board_time = 0

    def calculate_adjust_data(self):
        '''
        TODO:Test
        '''
        horizontal_board_projection_vector = subtraction_of_2_vector(self.top_project_position, self.position)
        uav_position = self.current_target.position
        uav_projection_point = [uav_position[0], uav_position[1], self.position[2]]
        uav_weapon_vector = subtraction_of_2_vector(uav_projection_point, self.position)

        self.end_position = add_of_2_vector(self.position, normalize(uav_weapon_vector))

        will_rotate_angle = cal_angle_of_2_vector(horizontal_board_projection_vector, uav_weapon_vector)
        horizontal_time = self._cal_time(will_rotate_angle)

        small = distance_of_2_point(self.position, uav_projection_point)
        large = distance_of_2_point(self.position, uav_position)
        uav_theta = radian_2_angle(acos_(small / large))
        self.end_theta = abs(uav_theta) if uav_position[2] >= self.position[2] else -abs(uav_theta)
        vertical_time = self._cal_time(abs(self.top_angle_with_xoy - self.end_theta))

        adjust_time = max(horizontal_time, vertical_time)
        logger.info("计算一次调弦时间：{}".format(adjust_time))
        self.need_adjust_board_time = adjust_time

    def build_dict(self):
        return self.config.build_dict()

    def set_search_rader(self, search_rader):
        self.search_rader = search_rader

    def get_current_target(self):
        '''
        当前目标必须不为空才调用这个方法
        :return:
        '''
        assert self.current_target is not None, "current target is None"
        return self.current_target

    def try_get_current_target(self):
        '''
        当前目标可以为空就调用这个方法
        :return:
        '''
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
        '''
        雷达的目标移除
        :return:
        '''
        if self.current_target is None:
            logger.info("移除目标时，当前目标不存在")
            assert False
            return
        self.current_target.reset_attacked_state()
        contain_list = []
        n = len(weapon.fired_bullet_list)
        for bullet in weapon.fired_bullet_list:
            if bullet.target is not self.current_target:
                contain_list.append(bullet)
        weapon.fired_bullet_list = contain_list
        logger.info(
            f"没有目标了就移除子弹，子弹不会执行step_attack_a_target_and_is_kill方法，移除前有子弹 {n}, 移除后 {len(weapon.fired_bullet_list)}")
        self.current_target = None

    def confirm_track_target(self, search_rader, uav_list, get_uav_index_fun):
        '''
        :param search_rader:
        :param uav_list:
        :return:
        '''
        # 计算威胁级别和判定范围
        threat_level_uav_list = self._adjust_board_prepare(search_rader, uav_list, get_uav_index_fun)
        if 0 == len(threat_level_uav_list):
            logger.info("没有搜索到或者没有在跟踪范围内")
            return None
        else:
            logger.info("搜索到或者在跟踪范围内")
        # （ entry[0]: 威胁程度，entry[1]: 对应无人机）
        max_threat_level = threat_level_uav_list[0][0]
        self.current_target = threat_level_uav_list[0][1]
        threat_level_string = ["威胁程度：" + str(entry[0]) + ", 对应无人机id：" + get_uav_index_fun(entry[1]) + "; " for
                               entry in
                               threat_level_uav_list]
        logger.info("威胁程度概述：" + str(threat_level_string), is_in_file=False)
        for entry in threat_level_uav_list:
            # is_block = self.map.judge_mountain(*self.position, *entry[1].position, 0, 'block')
            is_in = self.search_rader.is_a_uav_in_search_range(entry[1])
            # if is_block:
            #     logger.info(f"当前无人机因为被山遮挡从而没有被锁定")
            if max_threat_level < entry[0] and is_in:
                max_threat_level = entry[0]
                self.current_target = entry[1]
        if True:
            logger.info("找到了最大威胁程度大于阈值 " + str(

                GameConfig.THREAT_LEVEL_THRESHOLD) + "的无人机：" + self.current_target.print_self())
            assert self.current_target is not None, "_handle_equal_condition function exception"
        else:
            logger.info("没有找到，因为最大威胁程度小于阈值 " + str(GameConfig.THREAT_LEVEL_THRESHOLD))
            self.current_target = None

        for uav in threat_level_uav_list:
            logger.info(f"当前无人机和雷达距离  {compute_distance(uav[1].position, self.position)}")
        logger.info(f"选中的目标的距离为 {compute_distance(self.current_target.position, self.position)}")
        return self.current_target

    def adjust_board(self, fun):
        '''
        调整角度
        :return: True代表调舷成功
        '''
        self._step_adjust_board_one_time()
        end = self.need_adjust_board_time <= 0
        if end:
            self.adjust_end_set_value()
            return True
        return False

    def _step_adjust_board_one_time(self):
        '''
        time--
        :return:
        '''

        self.need_adjust_board_time -= UNIT_TIME

    def _adjust_board_prepare(self, search_rader, uav_list, fun):
        '''
        :param search_rader:
        :return: 返回元组列表，元组内容为（威胁程度，对应无人机），返回为空代表没有搜索到或者在追踪范围之外
        '''
        # 检测是否有飞机进入追踪范围，然后确定并输出追踪目标
        uav_to_track_list = search_rader.get_mark_3_time_uav_list(uav_list)
        uav_to_track_list = [a_uav for a_uav in uav_to_track_list if self._is_a_uav_in_track_range(a_uav, fun)]
        threat_level_uav_list = [(self._cal_threat_level(a_uav), a_uav) for a_uav in uav_to_track_list]
        return threat_level_uav_list

    def _is_a_uav_in_track_range(self, a_uav, fun):
        logger.info(
            "最小跟踪距离为：" + str(self.minimum_track_distance) + "，跟踪最远距离为：" + str(
                self.track_distance) + "，密集阵与id为" + fun(a_uav) + "的无人机的距离：" + str(
                distance_of_2_point(self.position,
                                    a_uav.position)), is_in_file=False)
        return is_a_point_in_a_sphere(self.track_distance, self.position, a_uav.position) and is_a_point_out_a_sphere(
            self.minimum_track_distance, self.position, a_uav.position)

    def is_target_in_track_range(self, fun):
        assert self.current_target is not None, "current target is None"
        logger.info(
            "最小跟踪距离为：" + str(self.minimum_track_distance) + "，跟踪最远距离为：" + str(
                self.track_distance) + "，密集阵与id为" + fun(self.current_target) + "的无人机的距离：" + str(
                distance_of_2_point(self.position,
                                    self.current_target.position)), is_in_file=False)
        return is_a_point_in_a_sphere(self.track_distance, self.position,
                                      self.current_target.position) and is_a_point_out_a_sphere(
            self.minimum_track_distance, self.position, self.current_target.position)

    def is_target_can_use_because_no_mountain(self):
        assert self.current_target is not None, "current target is None"
        # is_block = self.map.judge_mountain(*self.position, *self.current_target.position, 0, 'block')
        # if is_block:
        #     logger.info(f"密集阵与id为{self.current_target.id}的无人机之间有障碍物，无法使用")
        is_in = self.search_rader.is_a_uav_in_search_range(self.current_target)
        return is_in

    def is_target_in_fire_range(self):
        assert self.current_target is not None, "current target is None"
        logger.info("最小开火距离为：" + str(self.minimum_fire_distance) + "，开火最远距离为：" + str(
            self.fire_distance) + "，密集阵与id为" + self.current_target.get_id() + "无人机的距离：" + str(
            distance_of_2_point(self.position,
                                self.current_target.position)))
        return is_a_point_in_a_sphere(self.fire_distance, self.position,
                                      self.current_target.position) and is_a_point_out_a_sphere(
            self.minimum_fire_distance, self.position, self.current_target.position)

    def _cal_threat_level(self, single_uav):
        from numpy import Inf
        '''
        :param single_uav:
        :return: 计算威胁程度
        '''
        return cal_threat_level(self.position, single_uav.velocity, single_uav.velocity_direction, single_uav.position)

    def _cal_projection_velocity(self, single_uav):
        '''
        :return: 计算投影速度
        '''
        return cal_projection_velocity(single_uav, self.position)
