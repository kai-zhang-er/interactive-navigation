import numpy as np

from examples.pybullet.utils.pybullet_tools.pr2_problems import create_pr2
from examples.pybullet.utils.pybullet_tools.pr2_utils import set_arm_conf, arm_conf, REST_LEFT_ARM, close_arm, \
    set_group_conf, ARM_NAMES
from examples.pybullet.utils.pybullet_tools.utils import GREEN, add_data_path, load_pybullet, set_point, Point, create_box, \
    stable_z, HUSKY_URDF, dump_body, wait_for_user, GREY, BLACK, RED, BLUE, BROWN, TAN
from examples.pybullet.tamp.problems import sample_placements

from examples.pybullet.turtlebot_rovers.problems import RoversProblem


def problem1(n_rovers=1, n_objectives=1, n_rocks=2, n_soil=2, n_stores=1):
    room_size = 4.0
    base_limits = (-6,-2,6,2)
    corrido_w= 4.0
    mount_width = 0.5
    mound_height = 0.1

    #two room and a corridor
    floor = create_box(room_size*2+corrido_w, room_size, 0.001, color=TAN) 
    set_point(floor, Point(z=-0.001/2.))
    #horrizon walls
    wall1 = create_box(room_size*2+corrido_w + mound_height, mound_height, mound_height, color=GREY)
    set_point(wall1, Point(y=-room_size/2, z=mound_height/2.))
    wall2 = create_box(room_size + mound_height, mound_height, mound_height, color=GREY)
    set_point(wall2, Point(x=-4, y=room_size/2., z=mound_height/2.))
    wall3 = create_box(room_size + mound_height, mound_height, mound_height, color=GREY)
    set_point(wall3, Point(x=4.,y=room_size/2., z=mound_height/2.))
    wall4 = create_box(room_size + mound_height, mound_height, mound_height, color=GREY)
    set_point(wall4, Point(z=mound_height/2.))
    # vertical walls
    wall5 = create_box(mound_height, room_size + mound_height, mound_height, color=GREY)
    set_point(wall5, Point(x=-6., z=mound_height/2.))
    wall6 = create_box(mound_height, room_size + mound_height, mound_height, color=GREY)
    set_point(wall6, Point(x=6., z=mound_height/2.))
    wall7 = create_box(mound_height, room_size/2 + mound_height, mound_height, color=GREY)
    set_point(wall7, Point(x=-2.,y=1, z=mound_height/2.))
    wall8 = create_box(mound_height, room_size/2 + mound_height, mound_height, color=GREY)
    set_point(wall8, Point(x=2.,y=1, z=mound_height/2.))

    add_data_path()
    lander = load_pybullet(HUSKY_URDF, scale=1)
    lander_z = stable_z(lander, floor)
    set_point(lander, Point(-4, 1, lander_z))
    #wait_for_user()

    box_width = 0.5
    box_height = 1.0
    mound1 = create_box(box_width, box_width, box_height, color=GREEN)
    set_point(mound1, [-3, 0, box_height/2.])
    mound2 = create_box(mount_width, mount_width, mound_height, color=GREEN)
    set_point(mound2, [-3, -1, box_height/2.])

    initial_surfaces = {}

    rover_confs = [(-4, -1, 0)]
    assert n_rovers <= len(rover_confs)

    #body_names = map(get_name, env.GetBodies())
    landers = [lander]
    stores = ['store{}'.format(i) for i in range(n_stores)]

    #affine_limits = aabb_extrema(aabb_union([aabb_from_body(body) for body in env.GetBodies()])).T
    rovers = []
    for i in range(n_rovers):
        robot = create_pr2()
        dump_body(robot) # camera_rgb_optical_frame
        rovers.append(robot)
        set_group_conf(robot, 'base', rover_confs[i]) #set the position
        for arm in ARM_NAMES:
            set_arm_conf(robot, arm, arm_conf(arm, REST_LEFT_ARM))
            close_arm(robot, arm)

    obj_width = 0.07
    obj_height = 0.2

    objectives = []
    for _ in range(n_objectives):
        body = create_box(obj_width, obj_width, obj_height, color=BLUE)
        objectives.append(body)
        initial_surfaces[body] = mound1
    for _ in range(n_objectives):
        body = create_box(obj_width, obj_width, obj_height, color=RED)
        initial_surfaces[body] = mound2

    # TODO: it is moving to intermediate locations attempting to reach the rocks
    rocks = []
    for _ in range(n_rocks):
        body = create_box(0.075, 0.075, 0.05, color=BLACK)
        rocks.append(body)
        initial_surfaces[body] = floor
    soils = []
    for _ in range(n_soil):
        body = create_box(0.1, 0.1, 0.025, color=BROWN)
        soils.append(body)
        initial_surfaces[body] = floor
    sample_placements(initial_surfaces)

    #for name in rocks + soils:
    #    env.GetKinBody(name).Enable(False)
    return RoversProblem(rovers, landers, objectives, rocks, soils, stores, base_limits)

PROBLEMS = [
    problem1,
]