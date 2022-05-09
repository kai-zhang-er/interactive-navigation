import numpy as np
import gym
from gym import spaces
import pybullet as p
from env import create_env, test_collide, get_ray_info_around_point, \
    update_robot_base, test_drake_base_motion
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

        self.reward_dict={"collision":-3, "done":3, "distance_reward":-1}

        # actions for base
        self.action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # observation space
        self.observation_space=spaces.Box(low=0, high=1, shape=(num_rays,), dtype=np.float32)

        connect(use_gui=use_gui)
        set_camera_pose(camera_point=[0,-5,5], target_point=[0,0,0])
        with HideOutput():
            rover_confs=[self.init_pos, self.goal_pos]
            self.robots, self.movable=create_env(rover_confs=rover_confs)

        self.reward_list=[]

    def reset(self):
        # reset the robot to initial position
        self.robot_pos[:]=self.init_pos # assign value 
        ray_results_array=get_ray_info_around_point([self.robot_pos])
        observation=ray_results_array[:,2]
        return observation

    def step(self, action):

        reward=0
        if len(action)==2:
            sub_goal=self.robot_pos[:]+np.array([action[0],action[1],0],dtype=np.float32)
            # if test_drake_base_motion(self.robots[0], self.robot_pos, base_goal):
            sub_goal[0]=np.clip(sub_goal[0],self.base_limit[0][0], self.base_limit[0][1])
            sub_goal[1]=np.clip(sub_goal[1],self.base_limit[1][0], self.base_limit[1][1])
            ray=p.rayTest(self.robot_pos, sub_goal)
            if ray[0][0]<0: #not collision
                self.robot_pos[:]=sub_goal
                # reward+=0.2  # finish the subgoal

        else:
            raise ValueError("Received invalid action={}".format(action))
        
        done=False
        # collision reward
        # if test_collide([self.robot_pos]):
        #     reward+=self.reward_dict["collision"]
        
        # success reward
        distance=np.linalg.norm(self.robot_pos-self.goal_pos)
        
        if distance<self.T_dis:
            reward+=self.reward_dict["done"] 
            done=True
            # self.reward_list=[0]

        # distance reward
        self.reward_dict["distance"]=-distance/self.whole_distance
        reward=reward+self.reward_dict["distance"]

        info={"robot_pos":self.robot_pos}
        ray_results_array=get_ray_info_around_point([self.robot_pos])
        observation=ray_results_array[:,2]

        # self.reward_list.append(reward)
        # mean_reward=sum(self.reward_list)/len(self.reward_list)
        return observation, reward, done, info

    def render(self):
        update_robot_base(self.robots[0], self.robot_pos)
        # wait_for_duration(0.1)

    def close(self):
        disconnect()
        pass

    def change_obstacle_point(self, new_point):
        set_point(self.movable[0], Point(x=new_point[0],y=new_point[1], z=new_point[2]))