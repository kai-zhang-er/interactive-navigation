import random
import math
import numpy as np
import gym
from gym import spaces
import pybullet as p
from env import create_env, test_collide, get_ray_info_around_point, \
    update_robot_base, get_pr2_yaw, update_rover_base
from utils.pybullet_tools.utils import connect, disconnect, set_camera_pose, \
    wait_for_user, HideOutput, wait_for_duration, set_point, Point


class NAMOENV(gym.Env):
    """
    custom environment for namo task
    """
    def __init__(self, init_pos=(-4,-1,0),
                goal_pos=(4,1,0),
                base_limit=([-6+0.2,6-0.2], [-2+0.2,2]),
                distance_threshold=0.3, 
                num_rays=24,
                use_gui=True):
        super().__init__()

        # robot initial pos
        self.init_pos=np.array(init_pos,dtype=np.float32)
        self.robot_pos=self.init_pos.copy()
        
        self.goal_pos=np.array(goal_pos,dtype=np.float32)
        self.whole_distance=np.linalg.norm(self.init_pos-self.goal_pos)

        self.base_limit=base_limit
        self.T_dis=distance_threshold

        self.reward_dict={"collision":-3, "done":40, "distance_reward":-1}

        # actions for base
        self.action_space=spaces.Box(low=np.array([-1,-np.pi/3.0]), high=np.array([1,np.pi/3.0]), shape=(2,), dtype=np.float32)

        # observation space
        self.observation_space=spaces.Box(low=0, high=1, shape=(num_rays,), dtype=np.float32)

        # render configuration
        connect(use_gui=use_gui)
        set_camera_pose(camera_point=[0,-5,5], target_point=[0,0,0])
        with HideOutput():
            robot_confs=[self.init_pos, self.goal_pos]
            rover_confs=[(+3, 1, np.pi)]
            self.rover_pos=[[rover_confs[0][0],rover_confs[0][1],0.]]
            self.robots, self.movable, self.rovers=create_env(robot_confs=robot_confs, rover_confs=rover_confs)

        self.pr2_yaw=get_pr2_yaw(self.robots[0])
        self.reward_list=[]

    def reset(self):
        # reset the robot to initial position
        self.robot_pos[:]=self.init_pos # assign value 
        ray_results_array=get_ray_info_around_point([self.robot_pos])
        observation=ray_results_array[:,2]
        return observation

    def step(self, action):
        # rover random move
        self.random_move()

        reward=0.0
        if len(action)==2:   # linear and angular viteness
            vx, va=action
            yaw_world=self.pr2_yaw+va
            offset_x=math.cos(yaw_world)*vx
            offset_y=math.sin(yaw_world)*vx

            sub_goal=self.robot_pos[:]+np.array([offset_x, offset_y, 0],dtype=np.float32)
            # if test_drake_base_motion(self.robots[0], self.robot_pos, base_goal):
            sub_goal[0]=np.clip(sub_goal[0],self.base_limit[0][0], self.base_limit[0][1])
            sub_goal[1]=np.clip(sub_goal[1],self.base_limit[1][0], self.base_limit[1][1])
            ray=p.rayTest(self.robot_pos, sub_goal)
            if ray[0][0]<0: #not collision
                self.robot_pos[:]=sub_goal
                self.pr2_yaw+=va
                reward+=0.2  # finish the subgoal
        else:
            raise ValueError("Received invalid action={}".format(action))
        
        done=False
        # collision reward
        # binary reward 
        # if test_collide([self.robot_pos]):
        #     reward+=self.reward_dict["collision"]

        # success reward
        distance=np.linalg.norm(self.robot_pos-self.goal_pos)
        
        if distance<self.T_dis:
            reward+=self.reward_dict["done"] 
            done=True
            # self.reward_list=[0]

        # distance reward
        self.reward_dict["distance"]=-distance
        reward=reward+self.reward_dict["distance"]

        info={"robot_pos":self.robot_pos}
        ray_results_array=get_ray_info_around_point([self.robot_pos])
        observation=ray_results_array[:,2]
        # collision distance reward
        # min_distance=observation.min()+0.0001
        # collision_reward= self.reward_dict["collision"] if min_distance<self.T_dis else -1.0/min_distance
        # reward+=collision_reward

        # self.reward_list.append(reward)
        # mean_reward=sum(self.reward_list)/len(self.reward_list)
        return observation, reward, done, info

    def render(self):
        pr2_base_conf=(self.robot_pos[0], self.robot_pos[1], self.pr2_yaw)
        update_robot_base(self.robots[0], pr2_base_conf)
        update_rover_base(self.rovers[0], self.rover_pos)
        # wait_for_duration(0.05)

    def close(self):
        disconnect()
        pass

    def change_obstacle_point(self, new_point):
        set_point(self.movable[0], Point(x=new_point[0],y=new_point[1], z=new_point[2]))

    def random_move(self):
        # generate random number [-1,1]
        random_offset_x=random.random()*2-1
        random_offset_y=random.random()*2-1
        goal_pos=[self.rover_pos[0][0]+random_offset_x, self.rover_pos[0][1]+random_offset_y, 0.0]
        ray=p.rayTest(self.rover_pos[0], goal_pos)
        if ray[0][0]<0: #not collision
            self.rover_pos[0]=goal_pos

        

