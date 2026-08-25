from onpolicy.utils.math_tool import angle_2_radian, radian_2_angle


class AbstractComponentConfig:
    def __init__(self, **common_config):
        '''
        添加公有属性
        :param common_config:
        '''
        self.common_config = common_config
        for key in common_config:
            setattr(self, key, common_config[key])


class RaderConfig(AbstractComponentConfig):
    def __init__(self, **common_config):
        super().__init__(**common_config)


class SearchRaderConfig(RaderConfig):
    def __init__(self, **common_config):
        super().__init__(**common_config)
        self.detection_distance = 5.12 * 1000
        self.detection_probability = 0.99
        # 搜索确认时间，类型暂时为列表
        self.search_confirm_time = [0.6, 1.2, 1.8]

    def build_dict(self):
        return {
            "search_time": 0.6,
            "detection_probability": self.detection_probability,
            "detection_distance": self.detection_distance,
        }


class TrackRaderConfig(RaderConfig):
    def __init__(self, **common_config):
        super().__init__(**common_config)
        self.track_distance = 1829
        self.fire_distance = 1500
        self.minimum_track_distance = 0
        self.minimum_fire_distance = 2
        self.capture_time = 0
        self.capture_probability = 0.99
        self.rotate_angular_velocity = radian_2_angle(2)
        self.rotate_angular_acceleration = radian_2_angle(11)

    def get_track_distance(self):
        return self.track_distance

    def get_fire_distance(self):
        return self.fire_distance

    def build_dict(self):
        return {
            "capture_probability": self.capture_probability,
            "track_distance": self.track_distance,
            "capture_time": 0.4,
            "fire_time": 0.1,
        }


class WeaponConfig(AbstractComponentConfig):
    def __init__(self, **common_config):
        '''
            self.bullet_load_speed:（单位：发/分钟）
        '''
        super().__init__(**common_config)
        self.BULLET_CAPACITY = 1000
        self.bullet_load_speed = 400
        self.bullet_fire_speed = 75
        self.bullet_velocity = 1000
        self.hit_kill_probability = 1  # 0.005

    def build_dict(self):
        return {
            "bullet_fire_speed": self.bullet_fire_speed,
            "bullet_velocity": self.bullet_velocity,
            "bullet_shoot_distance": 1500,
            "bullet_capacity": self.BULLET_CAPACITY,
            "bullet_load_speed": self.bullet_load_speed,
        }
