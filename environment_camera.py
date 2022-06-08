import random
import math
import numpy as np
import gym
from gym import spaces
import pybullet as p
from skimage.draw import line_aa
from env import create_env, is_plan_possible, plan_pr2_base_motion, reset_movable_obstacles, get_ray_info_around_point, \
    update_robot_base, get_pr2_yaw, update_rover_base, update_movable_obstacle_point
from run import main_simple_version
from utils.pybullet_tools.utils import connect, disconnect, set_camera_pose, \
    wait_for_user, HideOutput, wait_for_duration, get_angle, get_distance, angle_between


class NAMOENV(gym.Env):
    """
    custom environment for namo task
    """
    def __init__(self, init_pos=(-4,-1,0),
                goal_pos=(4,1,0),
                base_limit=([-6+0.2,6-0.2], [-2+0.2,2]),
                distance_threshold=0.3,
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

        self.current_step=0
        self.max_steps_per_episode=1000

        self._COLLISION_THRESHOLD=0.2

        self._REMOVE_OBSTACLE_REWARD=-1
        self._NO_ACTION_REWARD=-1
        self._SUCCESS_REWARD=100
        self._DISTANCE_FACTOR_REWARD=10
        self._COLLISION_REWARD=-100

        # actions for base and arm
        # linear velocity, angular velocity, arm angle limit, arm length limit

        self.action_space=spaces.Box(low=np.array([0.,-np.pi/3.0]), high=np.array([1,np.pi/3.0]), shape=(2,), dtype=np.float32)

        # observation space
        _SCAN_RANGE_MIN=0.1
        _SCAN_RANGE_MAX=1.0
        self._N_RAYS=24
        _N_OBSERVATIONS=self._N_RAYS+4  # number of observations
        # low=np.ones((2,self._N_RAYS+1), dtype=np.float32)
        # low[0]=self._N_RAYS*[_SCAN_RANGE_MIN]+[0.0]
        # low[1]=self._N_RAYS*[-1]+[-math.pi]

        # high=np.ones((2,self._N_RAYS+1), dtype=np.float32)
        # high[0]=self._N_RAYS*[self._SCAN_RANGE_MAX]+[math.inf]
        # high[1]=self._N_RAYS*[20]+[math.pi]

        # self.observation_space=spaces.Box(low=low, high=high, shape=(2,self._N_RAYS+1), dtype=np.float32)
        # self.observation=np.ones((2,self._N_RAYS+1), dtype=np.float32)
        
        low=np.array(self._N_RAYS*[_SCAN_RANGE_MIN]+[0.0]+[-math.pi]+[self.base_limit[0][0], self.base_limit[1][0]])
        high=np.array(self._N_RAYS*[_SCAN_RANGE_MAX]+[math.inf]+[math.pi]+[self.base_limit[0][1], self.base_limit[1][1]])
        self.observation_space=spaces.Box(low=low, high=high, shape=(_N_OBSERVATIONS,), dtype=np.float32)

        # render configuration
        connect(use_gui=use_gui)
        set_camera_pose(camera_point=[0,-5,5], target_point=[0,0,0])
        with HideOutput():
            robot_confs=[self.init_pos, self.goal_pos]
            rover_confs=[(3, 1, np.pi)]
            self.rover_pos=[[rover_confs[0][0],rover_confs[0][1],0.]]
            self.robots, self.movable, self.rovers, self.statics=create_env(robot_confs=robot_confs, rover_confs=rover_confs)

        self.pr2_yaw=get_pr2_yaw(self.robots[0])


    def _update_observation2(self):
        self.ray_results_array=get_ray_info_around_point([self.robot_pos], ray_length= self.ray_length)
        obs_ray=self.ray_results_array[:,2]
        obs_info=self.ray_results_array[:,0]
        for i, o in enumerate(obs_info):
            if o in self.movable or o < 0:
                obs_info[i]=-1
            else:
                obs_info[i]=1

        distance=get_distance(self.robot_pos, self.goal_pos)
        angle=get_angle(self.robot_pos, self.goal_pos)
        angle_diff=angle-self.pr2_yaw
        self.observation[0,:-1]=obs_ray
        self.observation[0,-1]=distance
        self.observation[1,:-1]=obs_info
        self.observation[1,-1]=angle_diff
        # self.current_step+=1
        # print(self.current_step)
        return self.observation

    def _update_observation(self):
        self.ray_results_array=get_ray_info_around_point([self.robot_pos], ray_length= self.ray_length)
        obs_ray=self.ray_results_array[:,2]
        distance=get_distance(self.robot_pos, self.goal_pos)
        angle=get_angle(self.robot_pos, self.goal_pos)
        angle_diff=angle-self.pr2_yaw
        self.observation=np.concatenate((obs_ray, distance, angle_diff, self.robot_pos[:2]), axis=None)

    def _collision_occurred(self):
        # check collision with static or mobile obstacles
        return bool((self.observation[:-2]<self._COLLISION_THRESHOLD).any())

    def reset(self):
        self.current_step=0
        # reset obstacles to initial position
        reset_movable_obstacles(self.movable)
        # reset the robot to initial position
        self.robot_pos[:]=self.init_pos # assign value 
        self._update_observation()
        return self.observation

    def step(self, action):
        # rover random move
        self.random_move()
        self.pick=-1

        reward=0.0
        if len(action)==2:   # linear and angular viteness; arm angle and arm length
            vx, va=action
            yaw_world=self.pr2_yaw+va
            offset_x=math.cos(yaw_world)*vx
            offset_y=math.sin(yaw_world)*vx

            next_goal=self.robot_pos[:]+np.array([offset_x, offset_y, 0],dtype=np.float32)
            # if test_drake_base_motion(self.robots[0], self.robot_pos, base_goal):
            next_goal[0]=np.clip(next_goal[0],self.base_limit[0][0], self.base_limit[0][1])
            next_goal[1]=np.clip(next_goal[1],self.base_limit[1][0], self.base_limit[1][1])
            
            # if is_plan_possible(self.robots[0], next_goal, obstacles=self.statics):
            #     self.robot_pos[:]=next_goal
            #     self.pr2_yaw+=va
            # else:
            # angle=get_angle(self.robot_pos, next_goal)+math.pi
            # angle_interval=2*math.pi/self._N_RAYS
            # angle_index=round(yaw_world/angle_interval)%self._N_RAYS
            # obj_id=self.ray_results_array[angle_index, 0]
            # obj_id=self.observation[1, angle_index]

            self.pr2_yaw+=va 
            obj_id=p.rayTest(self.robot_pos, next_goal)[0][0]
            if obj_id<0 :
                self.robot_pos[:]=next_goal
            elif obj_id in self.movable:
                # when the movable obstacles block the way
                # remove it through pick-and-place
                # update_movable_obstacle_point(obj_id, (-3,1,0))
                self.pick=obj_id
                # reward+=self._REMOVE_OBSTACLE_REWARD
                self.robot_pos[:]=next_goal     
            else:
                reward=-0.1           
        else:
            raise ValueError("Received invalid action={}".format(action))
        
        done=False
        self._update_observation()
        # collision reward
        # if self._collision_occurred():
        #     reward+=self._COLLISION_REWARD
        # success reward
        if self.observation[-4]<self.T_dis:
            reward+=self._SUCCESS_REWARD 
            done=True
        # distance reward
        # dis_diff=self.whole_distance-self.observation[-4]
        # reward=reward+self._DISTANCE_FACTOR_REWARD*dis_diff

        info={"robot_pos":self.robot_pos}
        info.setdefault("pick", self.pick)
        return self.observation, reward, done, info

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
    def _lidar_to_occupancy_map(self):
        ray_results_array=get_ray_info_around_point([self.robot_pos], ray_length= self.ray_length)
        occupancy_map=np.zeros((20,20), dtype=np.uint8)
        for ray in ray_results_array:
            rr,cc,val=line_aa(self.robot_pos[:2], ray[2][:2])
            occupancy_map[rr,cc]=1
        return occupancy_map

        

