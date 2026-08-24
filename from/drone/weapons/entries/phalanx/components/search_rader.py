from onpolicy.envs.drone.weapons.entries.phalanx.components.rader import *


class SearchRader(Rader):
    def __init__(self, config, mmap):
        super().__init__(config)
        self.map = mmap
        self.search_interval = self.config.search_confirm_time[0]
        self.detection_distance = self.config.detection_distance
        self.position = config.common_config['position']

    def build_dict(self):
        return self.config.build_dict()

    def search_object(self, uav_list):
        # 检测是否有飞机进入搜索范围，注意下是实时检测
        self._search_policy(uav_list)

    def _search_policy(self, uav_list):
        '''
        累计搜索 or 连续搜索
        :return:
        '''
        if GameConfig.CURRENT_GAME_TIME * 10 % (self.search_interval * 10) == 0:
            # 每隔一定搜索频率标记无人机
            logger.info("标记了一下(只代表入口)")
            self._mark_uav_in_or_not_in_range(uav_list)

    def _mark_uav_in_or_not_in_range(self, uav_list):
        '''
        :param uav_list:
        :return:
        '''
        title = "标记情况"
        content = []
        for single_uav in uav_list:
            if single_uav is None:
                continue
            is_in_range = is_a_point_in_a_sphere(self.detection_distance, self.position, single_uav.position)

            if is_in_range:
                single_uav.search_marked_times += 1
                content_item = single_uav.print_self() + "的无人机被标记了一次，和密集阵的距离为" + str(
                    distance_of_2_point(self.position, single_uav.position))

            else:
                content_item = single_uav.print_self() + "的无人机因为距离为" + str(
                    distance_of_2_point(self.position, single_uav.position)) + "超出了范围，所以没有被标记成功"
            content.append(content_item)
            logger.info(content_item)
        logger.info(title, content)

    def is_a_uav_in_search_range(self, a_uav):
        return is_a_point_in_a_sphere(self.detection_distance, self.position, a_uav.position)

    def get_mark_3_time_uav_list(self, uav_list):
        '''
        :param uav_list:
        :return: 返回标记了三次的无人机列表
        '''
        uav_3_times_list = []
        for index, single_uav in enumerate(uav_list):
            if single_uav is None:
                logger.info("无人机{}被摧毁了，所以此无人机无法被列入搜索列表".format(index))
                continue

            # is_block = self.map.judge_mountain(*self.position, *single_uav.position, 0, 'block')

            if single_uav.search_marked_times >= TIME_TO_CONFIRM_A_UAV and \
                    self.is_a_uav_in_search_range(
                        single_uav):
                uav_3_times_list.append(single_uav)
            else:
                if not self.is_a_uav_in_search_range(single_uav):
                    logger.info(f"因为不在搜索范围被忽略")
                # if is_block:
                #     logger.info(f"因为山存在遮挡被忽略")
                if single_uav.search_marked_times < TIME_TO_CONFIRM_A_UAV:
                    logger.info(f"因为被标记次数不够被忽略")
        return uav_3_times_list
