from pprint import isreadable
import random
import math
import numpy as np
import gym
from gym import spaces
import pybullet as p
from env import create_env, is_plan_possible, plan_pr2_base_motion, reset_movable_obstacles, get_ray_info_around_point, \
    update_robot_base, get_pr2_yaw, update_rover_base, update_movable_obstacle_point, visualize_next_goal
from run import main_simple_version
from utils.pybullet_tools.utils import connect, disconnect, set_camera_pose, \
    wait_for_user, HideOutput, wait_for_duration, get_angle, get_distance, angle_between


class NAMOENV(gym.Env):
    """
    custom environment for namo task
    """
    def __init__(self, init_pos=(-4,-1,0),
                goal_pos=(4,1,0),
                base_limit=([-6+0.2,6-0.2], [-2+0.2,2-0.2]),
                distance_threshold=0.3,
                ray_length=1,
                use_gui=True):
        super().__init__()

        # robot initial pos
        self.init_pos=np.array(init_pos,dtype=np.float32)
        self.robot_pos=self.init_pos.copy()
        self.robot_current_pos=self.init_pos.copy()
        
        self.goal_pos=np.array(goal_pos,dtype=np.float32)
        self.whole_distance=np.linalg.norm(self.init_pos-self.goal_pos)

        self.base_limit=base_limit
        self.T_dis=distance_threshold

        self.ray_length=ray_length

        self.all_objects=None

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

        self.action_space=spaces.Box(low=np.array([0.,-np.pi/3.0]), high=np.array([0.8,np.pi/3.0]), shape=(2,), dtype=np.float32)

        # observation space
        _SCAN_RANGE_MIN=0.1
        _SCAN_RANGE_MAX=1.0
        self._N_RAYS=60
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
        set_camera_pose(camera_point=[0,-2,5], target_point=[0,0,0])
        with HideOutput():
            robot_confs=[self.init_pos, self.goal_pos]
            rover_confs=[(3, 1, np.pi)]
            self.rover_pos=[[rover_confs[0][0],rover_confs[0][1],0.]]
            self.robots, self.movable, self.rovers, self.statics=create_env(robot_confs=robot_confs, rover_confs=rover_confs)

        self.pr2_yaw=get_pr2_yaw(self.robots[0])

    def _update_observation(self):
        self.ray_results_array=get_ray_info_around_point([self.robot_pos], ray_length= self.ray_length, total_rays=self._N_RAYS)
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
        # reset obstacles to random position
        reset_movable_obstacles(self.movable)
        # reset the robot to ramdom position in starting region
        self.robot_pos[0]=np.random.random()*2-5  # -5<x<-3
        self.robot_pos[1]=np.random.random()*2-1  # -1<y<1
        self.goal_pos[0]=np.random.random()*2+3   # 3<x<5
        self.goal_pos[1]=np.random.random()*2-1  # -1<y<1
        self.goal_pos[2]=-0.1
        visualize_next_goal(self.statics[-3], self.goal_pos) # visualize goal
        visualize_next_goal(self.statics[-2], self.goal_pos) # visualize next subgoal
        self.whole_distance=np.linalg.norm(self.robot_pos-self.goal_pos)
        self.render() #update robot and rover render
        self._update_observation()

        return self.observation

    def step(self, action):
        # rover random move
        self.random_move()
        self.pick=[]
        self.all_objects=None

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
            # obj_id=p.rayTest(self.robot_pos, next_goal)[0][0]
            is_reachable, hit_movable_objects_list=self._test_reachable(next_goal)

            self.all_objects=set(self.ray_results_array[:,0].astype(int))-{-1, 0}
            if is_reachable:
                # when the next goal is reachable
                self.robot_current_pos[:]=self.robot_pos[:]
                self.robot_pos[:]=next_goal
                if len(hit_movable_objects_list)>0:
                    # when the movable obstacles block the way
                    # remove it through pick-and-place
                    # update_movable_obstacle_point(obj_id, (-3,1,0))
                    self.pick=hit_movable_objects_list
                    reward=reward-0.1
            else:
                reward=reward-0.5           
        else:
            raise ValueError("Received invalid action={}".format(action))
        
        done=False
        self._update_observation()
        # collision reward
        # if self._collision_occurred():
        #     reward+=self._COLLISION_REWARD
        # success reward
        if self.observation[self._N_RAYS]<self.T_dis:
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
        print("new step")
        visualize_next_goal(self.statics[-2], self.robot_pos) # next goal
        pr2_base_conf=(self.robot_pos[0], self.robot_pos[1], self.pr2_yaw)
        if len(self.pick)>0:
            # when there are movable obstacles to manipulate
            print(self.all_objects)
            print(self.pick)
            # stock region
            visualize_next_goal(self.statics[-1], self.robot_current_pos[:2]-0.5)
            working_limits={"0":[self.robot_current_pos[0]-self.ray_length, self.robot_current_pos[0]+self.ray_length],
                            "1":[self.robot_current_pos[1]-self.ray_length, self.robot_current_pos[1]+self.ray_length]}
            main_simple_version(self.pick, self, custom_limits=working_limits)
            # for obj in self.pick:
            #     main_simple_version([obj], self)
            print("finish grasp")
        is_plan,_=plan_pr2_base_motion(self.robots[0], pr2_base_conf, obstacles=self.all_objects)
        if not is_plan:
            self.robot_pos[:2]=self.robot_current_pos[:2]
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

    def _test_reachable(self, next_goal):
        robot_width=0.3  # consider the robot is a 0.6*0.6 square
        def get_bounding_box(center_point, radius):
            bounding_box=[]
            bounding_box.append([center_point[0]-radius, center_point[1]-radius, 0.])
            bounding_box.append([center_point[0]-radius, center_point[1]+radius, 0.])
            bounding_box.append([center_point[0]+radius, center_point[1]-radius, 0.])
            bounding_box.append([center_point[0]+radius, center_point[1]+radius, 0.])

            bounding_box.append([center_point[0]+radius, center_point[1], 0.])
            bounding_box.append([center_point[0]-radius, center_point[1], 0.])
            bounding_box.append([center_point[0], center_point[1]+radius, 0.])
            bounding_box.append([center_point[0], center_point[1]-radius, 0.])
            return bounding_box
        starting_points=get_bounding_box(self.robot_pos, robot_width)
        starting_points_array=np.array(starting_points)
        goal_points=get_bounding_box(next_goal, robot_width)
        goal_points_array=np.array(goal_points)
        ray_results=p.rayTestBatch(starting_points_array, goal_points_array, numThreads=2)
        hit_movable_object_list=[]
        is_reachable=True
        for ray in ray_results:
            if ray[0] in self.movable and ray[0] not in hit_movable_object_list:
                hit_movable_object_list.append(ray[0])
            elif ray[0] in self.statics:
                is_reachable=False
            
        return is_reachable, hit_movable_object_list

    def only_navigate(self):
        is_plan, base_path=plan_pr2_base_motion(self.robots[0], self.goal_pos, obstacles=self.statics[:-3]+self.movable)
        return is_plan, base_path

