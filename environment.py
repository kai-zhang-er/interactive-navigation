import numpy as np
import gym
from gym import spaces
from env import create_env, test_collide, get_ray_info_around_point, update_robot_base
from utils.pybullet_tools.utils import connect, disconnect, set_camera_pose, \
    wait_for_user, HideOutput, wait_for_duration, set_base_values


class NAMOENV(gym.Env):
    """
    custom environment for namo task
    """
    def __init__(self, init_pos=(-4,-1,0),
                goal_pos=(4,1,0),
                distance_threshold=0.3, 
                num_rays=24):
        super().__init__()

        # robot initial pos
        self.init_pos=np.array(init_pos,dtype=np.float32)
        self.robot_pos=self.init_pos
        self.goal_pos=np.array(goal_pos,dtype=np.float32)
        self.whole_distance=np.linalg.norm(self.init_pos-self.goal_pos)

        self.T_dis=distance_threshold

        self.reward_dict={"collision":-1, "done":1, "distance_reward":-1}

        # actions for base
        self.action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # observation space
        self.observation_space=spaces.Box(low=0, high=1, shape=(num_rays,), dtype=np.float32)

        connect(use_gui=True)
        set_camera_pose(camera_point=[0,-5,5], target_point=[0,0,0])
        with HideOutput():
            rover_confs=[self.init_pos, self.goal_pos]
            self.robots, movable=create_env(rover_confs=rover_confs)

    def reset(self):
        # reset the robot to initial position
        self.robot_pos=self.init_pos
        ray_results_array=get_ray_info_around_point([self.robot_pos])
        observation=ray_results_array[:,2]
        return observation

    def step(self, action):
        if len(action)==2:
            self.robot_pos+=np.array([action[0],action[1],0],dtype=np.float32)
        else:
            raise ValueError("Received invalid action={}".format(action))

        reward=0
        done=False
        # collision reward
        if test_collide([self.robot_pos]):
            reward+=self.reward_dict["collision"]
        
        # success reward
        distance=np.linalg.norm(self.robot_pos-self.goal_pos)
        
        if distance<self.T_dis:
            reward+=self.reward_dict["done"] 
            done=True

        # distance reward
        reward=reward-distance/self.whole_distance

        return self.robot_pos, reward, done

    def render(self):
        update_robot_base(self.robots[0], self.robot_pos)
        wait_for_duration(0.1)

    def close(self):
        disconnect()
        pass