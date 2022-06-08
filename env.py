import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import math
from utils.pybullet_tools.pr2_primitives import Conf

from utils.pybullet_tools.utils import connect, disconnect, draw_base_limits, get_angle, get_base_values, get_bodies, get_link_state, get_pose, plan_joint_motion, ray_from_pixel, wait_for_duration, \
    wait_for_user, get_image_at_pose, LockRenderer, joint_from_name, create_box, create_cylinder, HideOutput, GREY, TAN, RED, set_point,\
    Point, BLUE, base_values_from_pose, set_camera_pose, dump_body, euler_from_quat, load_model, TURTLEBOT_URDF, \
        joints_from_names, get_joint_positions, body_collision, angle_between
from utils.pybullet_tools.pr2_problems import create_pr2, Problem
from utils.pybullet_tools.pr2_utils import get_carry_conf, get_group_conf, get_other_arm, learned_forward_generator, load_inverse_reachability, open_arm, set_arm_conf, arm_conf, REST_LEFT_ARM, close_arm, \
    set_group_conf, ARM_NAMES, link_from_name, PR2_CAMERA_MATRIX, get_camera_matrix,PR2_GROUPS,\
        get_base_pose, set_joint_positions, get_disabled_collisions


TURTLE_BASE_JOINTS = ['x', 'y', 'theta']

def get_base_joints(robot):
    return joints_from_names(robot, TURTLE_BASE_JOINTS)

def get_base_conf(robot):
    return get_joint_positions(robot, get_base_joints(robot))

def set_base_conf(robot, conf):
    set_joint_positions(robot, get_base_joints(robot), conf)


def create_env(robot_confs = [(-4, -1, 0),(4,1,0)], n_robots=1, n_rovers=1, 
                rover_confs=[(+1, -1.75, np.pi)]):
    
    static_obstacles=create_walls()

    robots = []
    for i in range(n_robots):
        robot = create_pr2()
        # dump_body(robot) # camera_rgb_optical_frame
        robots.append(robot)
        set_group_conf(robot, 'base', robot_confs[i]) #set the position
        for arm in ARM_NAMES:
            set_arm_conf(robot, arm, arm_conf(arm, REST_LEFT_ARM))
            close_arm(robot, arm)
     
    rovers=add_mobile_obstacles(num=n_rovers, rover_confs=rover_confs)

    # TODO: make the objects smaller
    # cylinder_radius = 0.3
    # cylinder_height=1
    # body1 = create_cylinder(cylinder_radius, cylinder_height, color=RED)
    # set_point(body1, Point(x=-2,y=-0.5, z=0.))
    # body2 = create_cylinder(cylinder_radius, cylinder_height, color=BLUE)
    # set_point(body2, Point(-2,y=-1.5, z=0.))
    # movable = [body1, body2]

    movable_obstacles=create_movable_obstacles()
    # movable_obstacles=[]
    # print(get_bodies())
    return robots, movable_obstacles, rovers, static_obstacles

def create_walls():
    room_size = 4.0
    base_limits = (np.array([-6,-2]),np.array([6,2]))
    corrido_w= 4.0
    mound_height = 0.3

    # two room and a corridor
    # floor = create_box(room_size*2+corrido_w, room_size, 0.001, color=TAN) 
    # set_point(floor, Point(z=-0.001/2.))
    # horrizon walls
    wall1 = create_box(room_size*2+corrido_w + mound_height, mound_height, mound_height, color=GREY)
    set_point(wall1, Point(y=-room_size/2, z=0))
    wall2 = create_box(room_size*2+corrido_w + mound_height, mound_height, mound_height, color=GREY)
    set_point(wall2, Point(y=room_size/2, z=0))
    wall4 = create_box(corrido_w , 2, mound_height, color=GREY)
    set_point(wall4, Point(x=0.0, y=1, z=0))
    # vertical walls
    wall5 = create_box(mound_height, room_size + mound_height, mound_height, color=GREY)
    set_point(wall5, Point(x=-6., z=0.))
    wall6 = create_box(mound_height, room_size + mound_height, mound_height, color=GREY)
    set_point(wall6, Point(x=6., z=0.))
    wall7 = create_box(mound_height, room_size/2 + mound_height, mound_height, color=GREY)
    set_point(wall7, Point(x=-2.,y=1, z=0.))
    wall8 = create_box(mound_height, room_size/2 + mound_height, mound_height, color=GREY)
    set_point(wall8, Point(x=2.,y=1, z=0.))

    store_region = create_box(0.5, 0.5, 0.01, color=BLUE)
    set_point(store_region, Point(x=-3.,y=1, z=-0.1))

    goal_region= create_box(0.5, 0.5, 0.05, color=RED)
    set_point(goal_region, Point(x=4.,y=-1, z=-0.1))

    next_goal_marker=create_box(0.1, 0.1, 0.05, color=RED)

    static_obstacles=[wall1,wall2, wall4, wall5, wall6, wall7, wall8, next_goal_marker, store_region]
    return static_obstacles


def create_movable_obstacles():
    mass = 1
    movable_obstacles=[]
    # from (-2,0) to (-2,-2)
    
    cabbage = create_box(.1, .1, 0.8, mass=mass, color=(0, 1, 0, 1))
    set_point(cabbage, (-2, -0.5, 0))
    movable_obstacles.append(cabbage)

    cabbage2 = create_box(.1, .1, 0.8, mass=mass, color=(0, 1, 0, 1))
    set_point(cabbage2, (-2, -1.0, 0))
    movable_obstacles.append(cabbage2)

    cabbage3 = create_box(.1, .1, 0.8, mass=mass, color=(0, 1, 0, 1))
    set_point(cabbage3, (-2, -1.5, 0))
    movable_obstacles.append(cabbage3)

    return movable_obstacles

def reset_movable_obstacles(movable_obstacles):
    for i, m in enumerate(movable_obstacles):
        set_point(m, (-2, -0.5*(i+1), 0))

def namo_problem(arm='right', grasp_type='top'):
    other_arm = get_other_arm(arm)
    initial_conf = get_carry_conf(arm, grasp_type)

    # create pr2 abstract info
    pr2 = create_pr2()
    set_group_conf(pr2, 'base', (-1.5, -0.7, math.pi))
    set_arm_conf(pr2, arm, initial_conf)
    open_arm(pr2, arm)
    set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
    close_arm(pr2, other_arm)

    static_obstacles=create_walls()
    movable_obstacles=create_movable_obstacles()
    return Problem(pr2, movable=movable_obstacles, arms=[arm], 
                surfaces=[static_obstacles[-2]],
                sinks=static_obstacles,
                grasp_types=[grasp_type],
                goal_holding=[(arm, movable_obstacles[0])],
                goal_conf=get_pose(pr2),
                # goal_on=[(movable_obstacles[4], static_obstacles[-2])]
                )
    

def add_mobile_obstacles(num=1, rover_confs=[(+1, -1.75, np.pi)]):
    """add mobile obstacles

    Args:
        num (int, optional): number of obstacles to add. Defaults to 1.
        rover_confs (list, optional): (x, y, theta) position and orientation of obstacle
    
    return:
        a list of obstacles
    """
    assert num == len(rover_confs)
    rovers_list=[]
    for i in range(num):
        with HideOutput():
            rover = load_model(TURTLEBOT_URDF)
        set_point(rover, Point(z=0.))
        set_base_conf(rover, rover_confs[i])
        rovers_list.append(rover)
    return rovers_list

def get_pr2_yaw(robot_id):
    pose=get_base_pose(robot_id)
    _,_,yaw=euler_from_quat(pose[1])
    return yaw

def get_front_image_from_robot(robot_id):
    """get images from front camera

    Args:
        robot_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    width = 128
    height = 128

    head_name = 'high_def_optical_frame' # HEAD_LINK_NAME | high_def_optical_frame | high_def_frame
    head_link = link_from_name(robot_id, head_name)
    state=get_link_state(robot_id,head_link)
    pose=[(state[0][0],state[0][1], 0.4),state[1]]
    # camera_matrix=PR2_CAMERA_MATRIX
    camera_matrix=get_camera_matrix(width=width, height=height, fx=200, fy=200)
    images=get_image_at_pose(pose, camera_matrix)
    
    return images


def get_head_image_from_robot(robot_id):
    """get images from head camera

    Args:
        robot_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    width = 128
    height = 128

    head_name = 'high_def_optical_frame' # HEAD_LINK_NAME | high_def_optical_frame | high_def_frame
    head_link = link_from_name(robot_id, head_name)
    state=get_link_state(robot_id,head_link)
    pose=[(state[0][0],state[0][1], 0.5),state[1]]
    # camera_matrix=PR2_CAMERA_MATRIX
    camera_matrix=get_camera_matrix(width=width, height=height, fx=200, fy=200)
    images=get_image_at_pose(pose, camera_matrix)
    
    return images


def get_ray_info_around_robot(robot_id):
    """generate rays to detect surroundings

    Args:
        robot_id (int): robot id

    Returns:
        _type_: _description_
    """

    base_info = base_values_from_pose(get_base_pose(robot_id))
    ray_from_pos=np.array([[base_info[0], base_info[1],0.2]],dtype=np.float32)
    return get_ray_info_around_point(ray_from_pos)
    

def get_ray_info_around_point(ray_from_pos, ray_length=1):
    ray_from_pos=np.array(ray_from_pos, dtype=np.float32)
    total_rays=24
    angle_interval=2*math.pi/total_rays
    ray_from_pos_array=np.repeat(ray_from_pos, total_rays, axis=0)
    ray_to_pos_array=ray_from_pos_array.copy()
    for i in range(total_rays):
        delta_x=ray_length*math.cos(angle_interval*i)
        delta_y=ray_length*math.sin(angle_interval*i)
        ray_to_pos_array[i]+=np.array([delta_x, delta_y,0],dtype=np.float32)
    ray_results=p.rayTestBatch(ray_from_pos_array, ray_to_pos_array, numThreads=2)

    result_list=[]
    for r in ray_results:
        # distance=1000
        # if r[0] > -1: # hit something, 0 is floor
        #     distance=np.linalg.norm(ray_from_pos-r[3])
        result_list.append([r[0],r[1],r[2]*ray_length])
    return np.array(result_list,dtype=np.float32)

def test_passable(start_point, end_point):
    ray_width=1


def update_robot_base(robot_id, base_conf):
    """update the configuration of base joints

    Args:
        robot_id (_type_): robot id
        base_conf (_type_): (x,y,yaw))
    """
    set_group_conf(robot_id, 'base', base_conf)

def update_rover_base(rover_id, base_pos):
    base_conf=(base_pos[0][0], base_pos[0][1], np.pi)
    set_base_conf(rover_id, base_conf)

def update_movable_obstacle_point(obstacle_id, new_point):
    set_point(obstacle_id, Point(x=new_point[0], y=new_point[1], z=new_point[2]))

def visualize_next_goal(next_goal_marker, new_point):
    set_point(next_goal_marker, Point(x=new_point[0], y=new_point[1], z=-0.1))


def test_manipulate_region():
    arm_config=load_inverse_reachability("right", "top")
    print(arm_config)


def render_env(use_gui=True):
    simid=connect(use_gui=use_gui)
    set_camera_pose(camera_point=[0,-5,5], target_point=[0,0,0])

    with HideOutput():
        robots, movable, mobiles, statics=create_env()

    update_robot_base(robots[0], ((-3,0,0)))
    print(get_base_values(robots[0]))
    print(base_values_from_pose(get_base_pose(robots[0])))
    # images=get_front_image_from_robot(robots[0])
    # plt.figure()
    # plt.imshow(images[0])
    # plt.show()

    # set_point(movable[0], Point(x=-2.,y=1., z=0.))
    # ray_results=get_ray_info_around_robot(robots[0])
    # print(test_collide(np.array([[-5.0,-1.5,0.]])))
    # print(get_body_name(0))
    # print(plan_pr2_base_motion(robots[0], np.array([4.,1.,-1.0])))
    
    # test_ray()
    # get_pr2_yaw(robots[0])

    # set_point(movable[0], Point(x=-2.,y=2., z=0.))

    # for o in statics:
    #     if body_collision(movable[0], o):
    #         print("collide with {}".format(o))

    # update_robot_base(robots[0], base_conf=(4,1,np.pi))
    # wait_for_user()

    # test_manipulate_region()
    # t=learned_forward_generator(robots[0],(-3,0,0),"left", "top")
    # print(t)
    disconnect()

def test_ray():
    ray=p.rayTest(np.array([-4.0,-1.9, 0.]), np.array([0.,-2.1,0.]))
    print(ray[0][0]>0)

def plan_pr2_base_motion(robot_id, base_goal, obstacles=[]):
    disabled_collisions = get_disabled_collisions(robot_id)
    base_joints = [joint_from_name(robot_id, name) for name in PR2_GROUPS['base']]
    with LockRenderer(lock=True):
        base_path = plan_joint_motion(robot_id, base_joints, base_goal, obstacles=obstacles,
                                    disabled_collisions=disabled_collisions)
    if base_path is None:
        print('Unable to find a base path')
        print(base_goal)
        return
    # print("------------{}".format(base_path[0]))
    for bq in base_path:
        set_joint_positions(robot_id, base_joints, bq)
        wait_for_duration(0.005)

def is_plan_possible(robot_id, base_goal, obstacles=[]):
    disabled_collisions = get_disabled_collisions(robot_id)
    base_joints = [joint_from_name(robot_id, name) for name in PR2_GROUPS['base']]
    with LockRenderer(lock=True):
        base_path = plan_joint_motion(robot_id, base_joints, base_goal, obstacles=obstacles,
                                    disabled_collisions=disabled_collisions)
    if base_path is None:
        return False
    else:
        return True

if __name__=="__main__":
    render_env(use_gui=True)
    print(get_angle([0.,1.], [1, 0]))