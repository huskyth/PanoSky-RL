import os
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from onpolicy.utils.format_logger import AppLogger

logger = AppLogger().get_logger()


class Map(object):
    map = None

    @staticmethod
    def get_instance():
        if Map.map is None:
            Map.map = Map('N32E119.npz', 1, 1)
        return Map.map

    def __init__(self, map_data_file, map_resolution, uav_length):
        self.map_data_file = map_data_file  # 地图文件路径
        self.map_resolution = map_resolution  # 地图分辨率
        self.uav_length = uav_length  # 无人机长度

        map_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dem_data', self.map_data_file)
        data = np.load(map_data_path)

        k_x, k_y, k_z = ('data_x', 'data_y', 'data_z') if 'data_x' in data else ('arrx', 'arry', 'arrz')

        self.map_x = data[k_x]
        self.map_y = data[k_y] - min(data[k_y])  # 由于山地地图是从0递减的，所以现在减一个最小负值使之从正数递减
        self.map_z = data[k_z]
        self.map_max_x = max(self.map_x)
        self.map_min_x = min(self.map_x)
        self.map_max_y = max(self.map_y)
        self.map_min_y = min(self.map_y)
        self.map_x_dis = np.mean(self.map_x[1:] - self.map_x[:-1])  # 地图为等间隔，计算间隔值
        self.map_y_dis = np.mean(self.map_y[1:] - self.map_y[:-1])  # 地图为等间隔，计算间隔值
        self.map_x_start = self.map_x[0]  # 存0索引对应的位置x
        self.map_y_start = self.map_y[0]  # 存0索引对应的位置y

    def convert_index(self, p_x, p_y):
        diff_x = p_x - self.map_x_start  # 真实相差边界的距离
        index_x = int(diff_x // self.map_x_dis)  # 转换为索引x, 向下取整

        diff_y = p_y - self.map_y_start
        index_y = math.ceil(diff_y / self.map_y_dis)  # 转换为索引y, 向上取整

        return index_y, index_x  # map_z中数据为[y][x]格式存储，目前程序中读取均为按照[x][y]格式读取，交换index返回值以获取正确高度

    # search_nearest_height
    def search_nh(self, p_x, p_y):
        index_x, index_y = self.convert_index(p_x, p_y)
        try:
            return self.map_z[index_x, index_y]
        except IndexError:  # 如果索引超边界,在初始化无人机或者目标位置时候会遇到
            return -1000

    def judge_uav_ground_collision(self, p_x, p_y, p_z, coll_safe_dis):
        ground_heigh = self.search_nh(p_x, p_y)
        if p_z < ground_heigh + coll_safe_dis:
            return True  # 碰撞了
        return False  # 没撞

    def generate_img(self, p_x, p_y):
        # 生成127*127的，中心是无人机，左右各63
        index_x, index_y = self.convert_index(p_x, p_y)

        map_x_size, map_y_size = self.map_z.shape[0], self.map_z.shape[1]
        obs_size = 32  # 观测矩阵大小127*127
        obs_radius = obs_size // 2  # 以自身为中心后边缘大小

        obs_bottom_x = max(index_x - obs_radius, 0)
        obs_top_x = min(index_x + obs_radius, map_x_size)
        obs_bottom_y = max(index_y - obs_radius, 0)
        obs_top_y = min(index_y + obs_radius, map_y_size)

        observation = self.map_z[obs_bottom_x:obs_top_x, obs_bottom_y:obs_top_y]
        observation = np.pad(observation,
                             ((obs_radius - (index_x - obs_bottom_x), obs_radius - (obs_top_x - index_x)),
                              (obs_radius - (index_y - obs_bottom_y), obs_radius - (obs_top_y - index_y))),
                             'constant')

        observation = observation.reshape(1, 1, 32, 32)

        return observation

    def calculate_distance(self, x1, y1, z1, x2, y2, z2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distance

    def judge_mountain(self, x1, y1, z1, x2, y2, z2, coll_safe_dis, judge_type):
        logger.info(f"当前起点 {x1, y1, z1}, 终点 {x2, y2, z2}， 判断类型 {judge_type} 安全距离 {coll_safe_dis}，便于复现")
        if judge_type not in ['block', 'ground']:
            raise Exception("judge_type must be 'block' or 'ground'")
        distance = self.calculate_distance(x1, y1, z1, x2, y2, z2)
        interval = min(abs(self.map_x_dis), abs(self.map_y_dis))  # 分辨率取x，y间隔小的那个
        num_points = int(distance / interval)
        start = 1
        end = num_points if judge_type == 'block' else num_points + 1

        if num_points == 0 and judge_type == 'ground':
            return self.judge_uav_ground_collision(x2, y2, z2, coll_safe_dis)

        for i in range(start, end):  # 包括了初始末端两点，调用这个也可以免去调用单点的碰撞检测
            ratio = i / num_points
            x = x1 + (x2 - x1) * ratio
            y = y1 + (y2 - y1) * ratio
            z = z1 + (z2 - z1) * ratio
            if self.judge_uav_ground_collision(x, y, z, coll_safe_dis):
                return True  # 碰撞了

        return False

    @staticmethod
    def in_range(n, start, end=0):
        return start <= n <= end if end >= start else end <= n <= start

    @staticmethod
    def compute_dis(a, b):
        # 计算两点间距离
        dis = ((b[0] - a[0]) ** 2 +
               (b[1] - a[1]) ** 2
               ) ** 0.5
        return dis

    def get_map_boundary(self):
        x_index = self.map_x.argmax()
        y_index = self.map_y.argmax()
        x_max = self.map_x[x_index]
        y_max = self.map_y[y_index]
        return x_max, y_max

    def for_html(self, px, py):
        """
        为了前端页面的坐标适配
        """
        WIDTH = self.map_max_x - self.map_min_x
        HEIGHT = self.map_max_y - self.map_min_y
        assert WIDTH > 0 and HEIGHT > 0
        x = (px - WIDTH / 2) / abs(self.map_x_dis)
        y = (HEIGHT / 2 - py) / abs(self.map_y_dis)
        return x, y

    def visualize_3d(self, real_x, real_y, real_z, point_name="目标点"):
        """
        3D可视化地形 + 目标坐标点
        :param real_x: 实际X坐标
        :param real_y: 实际Y坐标
        :param real_z: 实际Z高度
        :param point_name: 点的名称（用于图例）
        """
        # 创建3D画布
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --------------------- 1. 绘制山地地形网格 ---------------------
        X, Y = np.meshgrid(self.map_x, self.map_y)
        Z = self.map_z

        # 绘制地形曲面（半透明方便看点）
        surf = ax.plot_surface(
            X, Y, Z,
            cmap='terrain',  # 地形配色
            alpha=0.7,  # 透明度
            linewidth=0,
            antialiased=True
        )

        # --------------------- 2. 绘制目标3D坐标点 ---------------------
        ax.scatter(
            real_x, real_y, real_z,
            color='red',  # 点颜色
            s=100,  # 点大小
            marker='*',  # 星形标记
            label=f'{point_name} ({real_x:.1f}, {real_y:.1f}, {real_z:.1f})'
        )

        # --------------------- 3. 图表设置 ---------------------
        ax.set_title('无人机地形3D可视化', fontsize=15, pad=20)
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        ax.set_zlabel('Z 高度', fontsize=12)
        ax.legend(loc='best')
        plt.colorbar(surf, ax=ax, shrink=0.5, label='海拔高度')

        # 自动适配视角
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
        plt.show()

    def generate_occluded_points(self, coll_safe_dis=1.0, max_try=100):
        """
        100% 生成一对被山体遮挡的 3D 点（绝对不会报错）
        原理：降低高度 + 控制距离，强制让山体挡住视线
        """
        x_max, y_max = self.get_map_boundary()

        # 安全范围：缩小一点，避免边缘无效点
        x_min, y_min = 50, 50
        x_max = max(x_max - 50, x_min)
        y_max = max(y_max - 50, y_min)

        for _ in range(max_try):
            # --------------------- 随机起点 ---------------------
            x1 = np.random.uniform(x_min, x_max)
            y1 = np.random.uniform(y_min, y_max)
            ground1 = self.search_nh(x1, y1)
            if ground1 < 0:
                continue

            # 关键：飞得低一点，更容易被山体挡住！
            z1 = ground1 + np.random.uniform(5, 15)

            # --------------------- 生成近距离终点（更容易被挡） ---------------------
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(3000, 10000)  # 距离拉近，遮挡率暴增

            x2 = x1 + dist * np.cos(angle)
            y2 = y1 + dist * np.sin(angle)

            # 边界检查
            if not (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
                continue

            ground2 = self.search_nh(x2, y2)
            if ground2 < 0:
                continue

            z2 = ground2 + np.random.uniform(5, 15)

            # --------------------- 判断遮挡（必中） ---------------------
            if self.judge_mountain(x1, y1, z1, x2, y2, z2, coll_safe_dis, 'block'):
                return round(x1, 2), round(y1, 2), round(z1, 2), \
                       round(x2, 2), round(y2, 2), round(z2, 2)

        # 终极兜底：强制生成一个低高度点（100%遮挡）
        x1, y1 = 100, 100
        g1 = self.search_nh(x1, y1)
        z1 = g1 + 8

        x2, y2 = 150, 150
        g2 = self.search_nh(x2, y2)
        z2 = g2 + 8

        return x1, y1, z1, x2, y2, z2


if __name__ == '__main__':
    map = Map("N32E119.npz", 30, 3.05)
    # for i in range(len(map.map_z)):
    #     for j in range(len(map.map_z[i])):
    #         if map.map_z[i][j] > 200:
    #             print("[{}, {}]".format(i, j))
    # with open("map_z.txt", "w") as f:
    #     f.write(map.map_z)
    x, y = map.map_max_x, map.map_max_y
    # print(map.convert_index(x, y))
    # print(map.search_nh(x, y))
    # print(map.for_html(x, y))
    # map.visualize_3d(x, y, 100, "zero_point")

    print(map.search_nh(6753.876487134699, 3971.893276789433))
    assert False
    ax, ay, az, bx, by, bz = map.generate_occluded_points()
    print(ax, ay, az, bx, by, bz)
    c1x, c1y = map.for_html(ax, ay)
    print(c1x, c1y, az)
    c2x, c2y = map.for_html(bx, by)
    print(c2x, c2y, bz)
    az, bz = int(az), int(bz)
    # map.visualize_3d(ax, ay, az, "zero_point")
    # map.visualize_3d(bx, by, bz, "zero_point")
    entity_data = [
        # 武器站
        {
            "id": "WPN-001",
            "entityType": "weapon",  # 必填：weapon 或 target
            "x": c1x,
            "z": c1y,
            "altitude": az,
            "range": 60,  # 射程（米）
            "ammo": 120,  # 弹药量
            "azimuth": 45,  # 方位角
            "elevation": 15  # 俯仰角
        },

        # 目标
        {
            "id": "TGT-001",
            "entityType": "target",  # 必填
            "x": c2x,
            "z": c2y,
            "altitude": bz,
            "threatLevel": "高",  # 高/中/低/极高
            "targetType": "装甲车",  # 自定义类型
            "threatRange": 40  # 威胁范围（米）
        },

    ]
    import json

    with open(r'C:\Users\qq162\Desktop\UAVSaver\front_interface\pages\entity_data.json', 'w', encoding='utf-8') as f:
        json.dump(entity_data, f, ensure_ascii=False, indent=2)
