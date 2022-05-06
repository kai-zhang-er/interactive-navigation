import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import math

from utils.pybullet_tools.utils import connect, disconnect, draw_base_limits, WorldSaver, get_link_state, quat_from_euler, ray_from_pixel, \
    wait_for_user, get_image_at_pose, LockRenderer, joint_from_name, create_box, create_cylinder, HideOutput, GREY, TAN, RED, set_point,\
         Point, BLUE, base_values_from_pose, set_camera_pose, dump_body, euler_from_quat
from utils.pybullet_tools.pr2_problems import create_pr2
from utils.pybullet_tools.pr2_utils import set_arm_conf, arm_conf, REST_LEFT_ARM, close_arm, \
    set_group_conf, ARM_NAMES, link_from_name, PR2_CAMERA_MATRIX, get_camera_matrix,PR2_GROUPS,\
        get_base_pose, set_joint_positions


def create_env(rover_confs = [(-4, -1, 0),(4,1,0)], n_rovers=1):
    room_size = 4.0
    base_limits = (np.array([-6,-2]),np.array([6,2]))
    corrido_w= 4.0
    mound_height = 0.2

    # two room and a corridor
    floor = create_box(room_size*2+corrido_w, room_size, 0.001, color=TAN) 
    set_point(floor, Point(z=-0.001/2.))
    # horrizon walls
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

    robots = []
    for i in range(n_rovers):
        robot = create_pr2()
        dump_body(robot) # camera_rgb_optical_frame
        robots.append(robot)
        set_group_conf(robot, 'base', rover_confs[i]) #set the position
        for arm in ARM_NAMES:
            set_arm_conf(robot, arm, arm_conf(arm, REST_LEFT_ARM))
            close_arm(robot, arm)
     

    goal_confs = {robots[0]: rover_confs[-1]}
    #goal_confs = {}

    # TODO: make the objects smaller
    cylinder_radius = 0.25
    cylinder_height=1
    body1 = create_cylinder(cylinder_radius, cylinder_height, color=RED)
    set_point(body1, Point(x=-3.3,y=-1., z=0.))
    body2 = create_cylinder(cylinder_radius, cylinder_height, color=BLUE)
    set_point(body2, Point(x=3.,y=1., z=0.))
    movable = [body1, body2]
    #goal_holding = {robots[0]: body1}
    goal_holding = {}
    return robots, movable 


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
    pose=[(state[0][0],state[0][1], 0.5),state[1]]
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
    ray_from_pos=np.array([[base_info[0], base_info[1],0.3]],dtype=np.float32)
    return get_ray_info_around_point(ray_from_pos)
    

def get_ray_info_around_point(ray_from_pos):
    total_rays=24
    ray_length=1.0
    angle_interval=2*math.pi/total_rays
    ray_from_pos_array=np.repeat(ray_from_pos, total_rays, axis=0)
    ray_to_pos_array=ray_from_pos_array.copy()
    for i in range(total_rays):
        delta_x=ray_length*math.cos(angle_interval*i)
        delta_y=ray_length*math.sin(angle_interval*i)
        ray_to_pos_array[i]+=np.array([delta_x, delta_y,0],dtype=np.float32)
    ray_results=p.rayTestBatch(ray_from_pos_array, ray_to_pos_array, numThreads=1)

    result_list=[]
    for r in ray_results:
        result_list.append([r[0],r[1],r[2]])
    return np.array(result_list,dtype=np.float32)


def test_collide(ray_from_pos):
    ray_results=get_ray_info_around_point(ray_from_pos)
    min_dist=1
    for r in ray_results:
        if r[0] !=-1 : # hit object id
            min_dist= r[2] if r[2]<min_dist else min_dist
    if min_dist<0.5: # distance threshold, smaller=>collide
        return True
    return False

def update_robot_base(robot_id, base_pos):
    base_joints = [joint_from_name(robot_id, name) for name in PR2_GROUPS['base']]
    set_joint_positions(robot_id, base_joints, base_pos)

def render_env():
    simid=connect(use_gui=True)
    set_camera_pose(camera_point=[0,-5,5], target_point=[0,0,0])

    with HideOutput():
        robots, movable=create_env()

    images=get_front_image_from_robot(robots[0])
    # plt.figure()
    # plt.imshow(images[0])
    # plt.show()
    # set_point(movable[0], Point(x=-2.,y=1., z=0.))
    ray_results=get_ray_info_around_robot(robots[0])
    
    wait_for_user()
    disconnect()

if __name__=="__main__":
    render_env()