import random
import math
import numpy as np
import gym
from gym import spaces
import pybullet as p
from env import create_env, is_plan_possible, plan_pr2_base_motion, reset_movable_obstacles, test_collide, get_ray_info_around_point, \
    update_robot_base, get_pr2_yaw, update_rover_base, update_movable_obstacle_point
from run import main_simple_version
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
                ray_length=1,
                use_gui=True):
        super().__init__()

        # robot initial pos
        self.init_pos=np.array(init_pos,dtype=np.float32)
        self.robot_pos=self.init_pos.copy()
        
        self.goal_pos=np.array(goal_pos,dtype=np.float32)
        self.whole_distance=np.linalg.norm(self.init_pos-self.goal_pos)

        self.base_limit=base_limit
        self.T_dis=distance_threshold

        self.ray_length=ray_length

        self.reward_dict={"collision":-3, "done":40, "distance_reward":-1}

        self.current_step=0
        self.max_steps_per_episode=1000

        # actions for base and arm
        # linear velocity, angular velocity, arm angle limit, arm length limit
        self.action_space=spaces.Box(low=np.array([-1,-np.pi/3.0, -np.pi/2.0, 0.3]), high=np.array([1,np.pi/3.0, np.pi/2.0, 1]), shape=(4,), dtype=np.float32)

        # observation space
        self.observation_space=spaces.Box(low=0, high=1, shape=(num_rays,), dtype=np.float32)

        # render configuration
        connect(use_gui=use_gui)
        set_camera_pose(camera_point=[0,-5,5], target_point=[0,0,0])
        with HideOutput():
            robot_confs=[self.init_pos, self.goal_pos]
            rover_confs=[(+3, 1, np.pi)]
            self.rover_pos=[[rover_confs[0][0],rover_confs[0][1],0.]]
            self.robots, self.movable, self.rovers, self.statics=create_env(robot_confs=robot_confs, rover_confs=rover_confs)

        self.pr2_yaw=get_pr2_yaw(self.robots[0])
        self.reward_list=[]

    def reset(self):
        self.current_step=0
        # reset obstacles to initial position
        reset_movable_obstacles(self.movable)
        # reset the robot to initial position
        self.robot_pos[:]=self.init_pos # assign value 
        ray_results_array=get_ray_info_around_point([self.robot_pos], ray_length= self.ray_length)
        observation=ray_results_array[:,2]
        return observation

    def step(self, action):
        # rover random move
        self.random_move()
        self.pick=-1

        reward=0.0
        if len(action)==4:   # linear and angular viteness; arm angle and arm length
            vx, va, aa, al=action 
            yaw_world=self.pr2_yaw+va
            offset_x=math.cos(yaw_world)*vx
            offset_y=math.sin(yaw_world)*vx

            sub_goal=self.robot_pos[:]+np.array([offset_x, offset_y, 0],dtype=np.float32)
            # if test_drake_base_motion(self.robots[0], self.robot_pos, base_goal):
            sub_goal[0]=np.clip(sub_goal[0],self.base_limit[0][0], self.base_limit[0][1])
            sub_goal[1]=np.clip(sub_goal[1],self.base_limit[1][0], self.base_limit[1][1])
            
            # if is_plan_possible(self.robots[0], sub_goal, obstacles=self.statics):
            #     self.robot_pos[:]=sub_goal
            #     self.pr2_yaw+=va
            # else:
            ray=p.rayTest(self.robot_pos, sub_goal)
            if ray[0][0]<0:
                self.robot_pos[:]=sub_goal
                self.pr2_yaw+=va
            elif ray[0][0] in self.movable:
                # when the movable obstacles block the way
                # remove it through pick-and-place
                arrange_angle=aa+self.pr2_yaw
                arrange_offset_x=math.cos(arrange_angle)*al
                arrange_offset_y=math.sin(arrange_angle)*al
                new_point=self.robot_pos[:]+np.array([arrange_offset_x, arrange_offset_y, 0],dtype=np.float32)
                if not test_collide([new_point], radius=1, threshold=0.1):
                    update_movable_obstacle_point(ray[0][0], new_point)
                    ray1=p.rayTest(self.robot_pos, sub_goal)
                    update_movable_obstacle_point(ray[0][0], ray[0][3])
                    if ray1[0][0]<0: # not collision
                        self.robot_pos[:]=sub_goal
                        self.pr2_yaw+=va
                        self.pick=ray[0][0]
                        # reward=-1
                        
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
        self.reward_dict["distance"]=math.log(self.whole_distance)-math.log(distance)
        reward=reward+self.reward_dict["distance"]

        info={"robot_pos":self.robot_pos}
        ray_results_array=get_ray_info_around_point([self.robot_pos], ray_length=self.ray_length)
        observation=ray_results_array[:,2]
        # collision distance reward
        # min_distance=observation.min()+0.0001
        # collision_reward= self.reward_dict["collision"] if min_distance<self.T_dis else -1.0/min_distance
        # reward+=collision_reward

        # self.reward_list.append(reward)
        # mean_reward=sum(self.reward_list)/len(self.reward_list)
        
        # maximum steps constraints
        # self.current_step+=1
        # if self.current_step>self.max_steps_per_episode:
        #     done=True
        info.setdefault("pick", self.pick)
        return observation, reward, done, info

    def render(self):
        pr2_base_conf=(self.robot_pos[0], self.robot_pos[1], self.pr2_yaw)
        update_robot_base(self.robots[0], pr2_base_conf)
        update_rover_base(self.rovers[0], self.rover_pos)
        # wait_for_duration(0.05)

    def render_steps(self):
        if self.pick>-1:
            main_simple_version(self.pick, self)
        pr2_base_conf=(self.robot_pos[0], self.robot_pos[1], self.pr2_yaw)
        plan_pr2_base_motion(self.robots[0], pr2_base_conf)
        # update_robot_base(self.robots[0], pr2_base_conf)
        update_rover_base(self.rovers[0], self.rover_pos)

    def close(self):
        disconnect()
        pass

    def random_move(self):
        # generate random number [-0.5,0.5]
        random_offset_x=random.random()-0.5
        random_offset_y=random.random()-0.5
        goal_pos=[self.rover_pos[0][0]+random_offset_x, self.rover_pos[0][1]+random_offset_y, 0.0]
        ray=p.rayTest(self.rover_pos[0], goal_pos)
        if ray[0][0]<0: #not collision
            self.rover_pos[0]=goal_pos

        

