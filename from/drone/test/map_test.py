from onpolicy.envs.drone.maps.map import Map
from onpolicy.utils.util import compute_distance

mountain = r'C:\Users\qq162\Desktop\PanoSky-RL\onpolicy\envs\drone\maps\dem_data\N32E119.npz'
mmap = Map(mountain, 30, 3.05)

position, t_position = (5374.637103666594, 9179.297066429024, 16.0), (
    4713.840152471758, 8936.881951847436, -2.4000301311309045)

print(compute_distance(position, t_position))
is_block = mmap.judge_mountain(*position, *t_position, coll_safe_dis=0, judge_type='block')

print(is_block)
