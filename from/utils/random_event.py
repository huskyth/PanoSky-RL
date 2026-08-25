from onpolicy.utils.math_tool import *


def single_probability_event(p):
    assert 0 <= p <= 1, 'probability range error'
    num = generate_a_random_number()
    return True if num < p else False


def generate_round_inner_point(input_point, max_radius):
    radius = generate_a_ranged_random_number(0, max_radius)
    ret = [0, 0, 0]
    alpha = generate_a_ranged_random_number(0, 360)
    dx, dy = radius * cos(angle_2_radian(alpha)), radius * sin(angle_2_radian(alpha))
    ret[0] = input_point[0] + dx
    ret[1] = input_point[1] + dy
    ret[2] = input_point[2]
    return ret


if __name__ == '__main__':
    set_seed(100)
    y = generate_a_ranged_random_number(0, 10000)
    print(y)
    y = generate_a_ranged_random_number(0, 10000)
    print(y)
    print(cos(angle_2_radian(60)))

    point = [0, 0, 0]
    generate_round_inner_point(point, 100)
    print(point)
