import random
from math import cos, sin, sqrt, pi, asin, acos, ceil

from onpolicy.envs.drone.weapons.entries.config.global_config import GameConfig
from onpolicy.utils.format_logger import AppLogger

logger = AppLogger().get_logger()


def set_seed(seed):
    return
    random.seed(seed)


def clip(from_number, to_number, number):
    assert from_number <= to_number, 'from cannot large than to'
    return min(max(number, from_number), to_number)


def ceil_number(number):
    return ceil(number)


def get_rotate_left_matrix_around_z(alpha):
    return [[cos(alpha), sin(alpha), 0], [-sin(alpha), cos(alpha), 0], [0, 0, 1]]


def get_rotate_left_matrix_around_x(alpha):
    return [[1, 0, 0], [0, cos(alpha), sin(alpha)], [0, -sin(alpha), cos(alpha)]]


def vector_type_check_tool(vector):
    assert isinstance(vector, list), "point is not a list"
    assert len(vector) != 0, "vector length cannot be 0"


def matrix_length_check_tool(m):
    assert len(m) != 0, "matrix row can not be 0"
    assert len(m[0]) != 0, "matrix column can not be 0"


def matrix_type_check_tool(m):
    vector_type_check_tool(m)
    for row_a in m:
        vector_type_check_tool(row_a)


def between_vectors_check_tool(vector1, vector2):
    vector_type_check_tool(vector1)
    vector_type_check_tool(vector2)
    assert len(vector1) == len(vector2), "point list length not equal"


def generate_a_random_number():
    GameConfig.RANDOM_N += 1
    return random.random()


def generate_a_ranged_random_number(from_value, to_value):
    GameConfig.RANDOM_N += 1
    return random.uniform(from_value, to_value)


def subtraction_of_2_vector(vector1, vector2):
    between_vectors_check_tool(vector1, vector2)
    n = len(vector1)
    subtraction_result = [0] * n
    for index in range(n):
        subtraction_result[index] = vector1[index] - vector2[index]
    return subtraction_result


def add_of_2_vector(vector1, vector2):
    between_vectors_check_tool(vector1, vector2)
    n = len(vector1)
    add_result = [0] * n
    for index in range(n):
        add_result[index] = vector1[index] + vector2[index]
    return add_result


def is_a_point_in_a_sphere(radius, sphere_center, out_point):
    '''
    :param radius: scalar
    :param sphere_center: list vector
    :param out_point: list vector
    :return:
    '''
    distance = distance_of_2_point(sphere_center, out_point)
    return distance < radius


def is_a_point_out_a_sphere(radius, sphere_center, out_point):
    distance = distance_of_2_point(sphere_center, out_point)
    return distance > radius


def distance_of_2_point(point_1, point_2):
    between_vectors_check_tool(point_1, point_2)
    distance = 0
    for index in range(len(point_1)):
        distance += pow(point_1[index] - point_2[index], 2)
    return sqrt(distance)


def length_of_vector(vector):
    between_vectors_check_tool(vector, vector)
    distance = 0
    for index in range(len(vector)):
        distance += pow(vector[index], 2)
    return sqrt(distance)


def dot_of_2_vector(vector1, vector2):
    between_vectors_check_tool(vector1, vector2)
    dot_result = 0
    for index in range(len(vector1)):
        dot_result += vector1[index] * vector2[index]
    return dot_result


def cal_angle_of_2_vector(vector1, vector2):
    '''
    :param vector1:
    :param vector2:
    :return: 返回以度为单位的夹角
    '''

    dot_value = dot_of_2_vector(vector1, vector2)
    length1 = length_of_vector(vector1)
    length2 = length_of_vector(vector2)
    a_cos_v = clip(-1, 1, dot_value / (length2 * length1))
    return radian_2_angle(acos(a_cos_v))


def acos_(rate):
    return acos(rate)


def radian_2_angle(radian):
    return 180 * radian / pi


def angle_2_radian(angle):
    return angle * pi / 180


def matrix_checker(m):
    matrix_type_check_tool(m)
    matrix_length_check_tool(m)


def matrix_mul_vector(matrix, vector):
    matrix_checker(matrix)
    vector_type_check_tool(vector)
    result_row = len(matrix)
    result = [0 for i in range(result_row)]
    for row in range(result_row):
        result[row] = dot_of_2_vector(matrix[row], vector)
    return result


def matrix_a_mul_b(a, b):
    matrix_checker(a)
    matrix_checker(b)
    a_row, a_colunm, b_row, b_column = len(a), len(a[0]), len(b), len(b[0])
    assert a_colunm == b_row, "matrix cannot mul"
    result = [[0 for i in range(b_column)] for j in range(a_row)]
    for i in range(a_row):
        for j in range(b_column):
            result[i][j] = dot_of_2_vector(a[i], b[j])
    return result


def scalar_mul_vector(scalar, vector):
    vector_type_check_tool(vector)
    result = []
    for x in range(len(vector)):
        result.append(vector[x] * scalar)
    return result


def normalize(vector):
    vector_type_check_tool(vector)
    distance = length_of_vector(vector)
    if distance == 0:
        return vector
    n_vector = len(vector)
    result = [0 for i in range(n_vector)]
    for x in range(n_vector):
        result[x] = vector[x] / distance
    return result


def transpose(m):
    matrix_checker(m)
    row, column = len(m), len(m[0])
    result = [[0 for i in range(row)] for j in range(column)]
    for i in range(row):
        for j in range(column):
            result[j][i] = m[i][j]
    return result


def cross_product(vector1, vector2):
    vector_type_check_tool(vector1)
    vector_type_check_tool(vector2)
    assert len(vector1) == 3, "cross product length can be 3"
    a1, a2, a3, b1, b2, b3 = vector1[0], vector1[1], vector1[2], vector2[0], vector2[1], vector2[2]
    r1 = a2 * b3 - a3 * b2
    r2 = b1 * a3 - a1 * b3
    r3 = a1 * b2 - b1 * a2
    return [r1, r2, r3]


def is_zero_vector(vector):
    vector_type_check_tool(vector)
    for x in vector:
        if x != 0:
            return False
    return True


def a_sin_angle(v):
    return radian_2_angle(asin(v))


def fly_from_9_selections(vertical_radian, horizontal_radian, velocity, horizontal_right_vector,
                          current_position):
    '''
    目前先水平后垂直
    角度传为弧度
    '''
    velocity_distance = length_of_vector(velocity)
    z_axis_in_parent = normalize(cross_product(velocity, horizontal_right_vector))
    y_axis_in_parent = normalize(velocity)
    x_axis_in_parent = normalize(cross_product(y_axis_in_parent, z_axis_in_parent))

    matrix_parent2child = [x_axis_in_parent, y_axis_in_parent, z_axis_in_parent]
    matrix_child2parent = transpose(matrix_parent2child)

    velocity_direction_in_child = matrix_mul_vector(matrix_parent2child, velocity)
    horizontal_right_vector_in_child = matrix_mul_vector(matrix_parent2child, horizontal_right_vector)

    after_rotate_velocity_direction_in_child = matrix_mul_vector(get_rotate_left_matrix_around_z(horizontal_radian),
                                                                 velocity_direction_in_child)
    after_rotate_velocity_direction_in_child = matrix_mul_vector(get_rotate_left_matrix_around_x(vertical_radian),
                                                                 after_rotate_velocity_direction_in_child)

    after_rotate_horizontal_right_vector_in_child = matrix_mul_vector(
        get_rotate_left_matrix_around_z(horizontal_radian),
        horizontal_right_vector_in_child)
    after_rotate_horizontal_right_vector_in_child = matrix_mul_vector(get_rotate_left_matrix_around_x(vertical_radian),
                                                                      after_rotate_horizontal_right_vector_in_child)

    after_rotate_velocity_direction_in_parent = matrix_mul_vector(matrix_child2parent,
                                                                  after_rotate_velocity_direction_in_child)
    after_rotate_horizontal_right_vector_in_parent = matrix_mul_vector(matrix_child2parent,
                                                                       after_rotate_horizontal_right_vector_in_child)

    fly_distance_in_parent = scalar_mul_vector(velocity_distance,
                                               normalize(after_rotate_velocity_direction_in_parent))

    current_position = add_of_2_vector(fly_distance_in_parent, current_position)
    after_rotate_velocity_direction_in_parent = normalize(after_rotate_velocity_direction_in_parent)
    after_rotate_horizontal_right_vector_in_parent = normalize(after_rotate_horizontal_right_vector_in_parent)

    return current_position, after_rotate_velocity_direction_in_parent, after_rotate_horizontal_right_vector_in_parent


def cal_projection_velocity(weapon_position, uav_velocity, uav_velocity_direction, uav_position):
    '''
    :return: 计算投影速度
    '''
    track_2_uav_vector = normalize(subtraction_of_2_vector(weapon_position, uav_position))

    v_ = dot_of_2_vector(track_2_uav_vector, scalar_mul_vector(uav_velocity, normalize(uav_velocity_direction)))
    logger.info(
        f"投影速度为 {v_}, uav_velocity = {uav_velocity}, uav_velocity_direction = {uav_velocity_direction},track_2_uav_vector = {track_2_uav_vector}")
    return v_


def cal_threat_level(weapon_position, uav_velocity, uav_velocity_direction, uav_position):
    distance = distance_of_2_point(uav_position, weapon_position)
    distance_variation = cal_projection_velocity(weapon_position, uav_velocity, uav_velocity_direction, uav_position)
    logger.info("uav_position = {}, self.position = {},distance_variation = {}".format(uav_position, weapon_position,
                                                                                       distance_variation))
    if distance == 0: return float('inf')
    return distance_variation / distance
