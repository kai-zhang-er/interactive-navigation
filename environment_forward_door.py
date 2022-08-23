import time
import random
import math
import cv2
import numpy as np
import gym
from gym import spaces
import pybullet as p
from env import convert_pos_to_imagepos, create_env2, generate_no_collision_pose, get_occupancy_map_from_pos, is_plan_possible, plan_pr2_base_motion, reset_door, reset_movable_obstacles, get_ray_info_around_point, reset_movable_obstacles2, \
    update_robot_base, get_pr2_yaw, update_rover_base, update_movable_obstacle_point, visualize_next_goal
# from run import main_simple_version
from utils.pybullet_tools.utils import connect, disconnect, set_camera_pose, \
    wait_for_user, HideOutput, wait_for_duration, get_angle, get_distance, angle_between


class NAMOENV(gym.Env):
    """
    custom environment for namo task
    """
    def __init__(self, init_pos=(-4,-1,0),
                goal_pos=(4,1,0),
                distance_threshold=1,
                ray_length=2,
                use_gui=True):
        super().__init__()

        # robot initial pos
        self.init_pos=np.array(init_pos,dtype=np.float32)
        self.robot_pos=self.init_pos.copy()
        self.robot_current_pos=self.init_pos.copy()
        
        self.goal_pos=np.array(goal_pos,dtype=np.float32)
        self.whole_distance=np.linalg.norm(self.init_pos-self.goal_pos)

        self.T_dis=distance_threshold

        self.ray_length=ray_length

        self.all_objects=None

        self.current_step=0

        self.reward_map_open_door=np.load("cost_map_hole_env.npy")/20
        self.reward_map_close_door=np.load("cost_map_easy_env.npy")/20
        self._SUCCESS_REWARD=0

        self.rover_pos=np.array([[2,0,0]])

        # self.global_map=np.zeros((300,300), dtype=np.uint8)
        # self.global_map=cv2.imread("simple_env.png")

        # actions for base and arm
        # linear velocity, angular velocity
        self.action_space=spaces.Box(low=np.array([-0.95,-np.pi/4]), high=np.array([0.95, np.pi/4]), shape=(2,), dtype=np.float32)

        # observation space
        _SCAN_RANGE_MIN=0.1
        _SCAN_RANGE_MAX=ray_length
        self._N_RAYS=60
        _N_OBSERVATIONS=self._N_RAYS+2  # number of observations        

        # render configuration
        connect(use_gui=use_gui)
        set_camera_pose(camera_point=[0,-2,10])
        with HideOutput():
            self.robots, self.statics, self.movable, self.mobile, self.base_limits=create_env2(config_txt="simple_env_hole.txt", num_movable_obstacles=10, with_door=True)
    
        self.pr2_yaw=get_pr2_yaw(self.robots[0])

        low=np.array(self._N_RAYS*[_SCAN_RANGE_MIN]+[0.0]+[-math.pi])
        high=np.array(self._N_RAYS*[_SCAN_RANGE_MAX]+[13.0]+[math.pi])
        
        self.observation_space=spaces.Box(low=low, high=high, shape=(_N_OBSERVATIONS,), dtype=np.float32)

    def _update_observation(self):
        self.ray_results_array=get_ray_info_around_point([self.robot_pos], ray_length= self.ray_length, total_rays=self._N_RAYS)
        obs_ray=self.ray_results_array[:,2]
        distance=get_distance(self.robot_pos, self.goal_pos)
        angle=get_angle(self.robot_pos, self.goal_pos)
        observation=np.concatenate((obs_ray, distance, angle), axis=None)
        self.observation=(observation-self.observation_space.low)/(self.observation_space.high-self.observation_space.low)*2-1
        # print("map shape: ", small_map.shape)

    def reset(self, random_pt=True):
        t=time.time()
        # save_fig_name="tmp/{}_{}.jpg".format(t//60, int(t%60))
        # cv2.imwrite(save_fig_name, (self.global_map*125).astype(np.uint8))
        # self.global_map=np.zeros((300,300), dtype=np.uint8)

        self.reward_map=self.reward_map_close_door
        # reset obstacles to random position
        reset_movable_obstacles2(self.movable[:-1], self.base_limits)
        reset_door(self.movable[-1], [2, 1.75])
        # reset the robot to ramdom position in starting region
        # 
        self.pr2_yaw=(random.random()*2-1)*np.pi


        self.robot_pos[:2]=generate_no_collision_pose([[-4,4],[-4,-2]])
        # self.robot_pos[:2]=np.array([-7,-5])
        # self.goal_pos[:2]=generate_no_collision_pose(self.base_limits)
        self.goal_pos[:2]=np.array([-0.5,4])
        

        self.goal_pos[2]=-0.1
        self.robot_current_pos[:]=self.robot_pos[:]
        self.goal_image_pos=convert_pos_to_imagepos(self.goal_pos, self.reward_map.shape)  #col, row
        # print("goal pos:", self.goal_image_pos)

        visualize_next_goal(self.statics[-2], self.goal_pos) # visualize goal
        visualize_next_goal(self.statics[-1], self.goal_pos) # visualize next subgoal
        self.whole_distance=np.linalg.norm(self.robot_pos-self.goal_pos)
        self.rover_pos[0,:2]=generate_no_collision_pose(self.base_limits)
        self.render() #update robot and rover render
        self._update_observation()
        return self.observation

    def step(self, action):
        # rover random move
        self.random_move()
        self.pick=[]
        self.all_objects=None

        action = np.clip(action, self.action_space.low, self.action_space.high)

        reward=0.0
        if len(action)==2:   # linear and angular viteness; arm angle and arm length
            vx, va=action
            # va=4*va
            yaw_world=self.pr2_yaw+va
            offset_x=math.cos(yaw_world)*vx
            offset_y=math.sin(yaw_world)*vx

            next_goal=self.robot_pos[:]+np.array([offset_x, offset_y, 0],dtype=np.float32)
            # if test_drake_base_motion(self.robots[0], self.robot_pos, base_goal):
            next_goal[0]=np.clip(next_goal[0],self.base_limits[0][0], self.base_limits[0][1])
            next_goal[1]=np.clip(next_goal[1],self.base_limits[1][0], self.base_limits[1][1])
                        
            self.pr2_yaw+=va 
            # obj_id=p.rayTest(self.robot_pos, next_goal)[0][0]
            is_reachable, hit_movable_objects_list=self._test_reachable(next_goal)

            self.all_objects=set(self.ray_results_array[:,0].astype(int))-{-1, 0}
            if is_reachable:
                # when the next goal is reachable
                self.robot_current_pos[:]=self.robot_pos[:]
                self.robot_pos[:]=next_goal
                # reward=reward-0.005 #step penalty
                
                if len(hit_movable_objects_list)>0:
                    # when the movable obstacles block the way
                    # remove it through pick-and-place
                    # update_movable_obstacle_point(obj_id, (-3,1,0))
                    self.pick=hit_movable_objects_list
                    # pick penalty
                    reward=reward-0.1
                if self.movable[-1] in hit_movable_objects_list:
                    reward=reward-0.2  # cost to open door
                    self.reward_map=self.reward_map_open_door
                    reset_door(self.movable[-1], [6, 6])
            else:
                # print("not reachable, {}->{}".format(self.robot_pos, next_goal))
                # reward=reward-0.05  
                # reward=-5
                pass         
        else:
            raise ValueError("Received invalid action={}".format(action))
        
        done=False
        self._update_observation()

        # distance reward
        next_img_pos=convert_pos_to_imagepos(self.robot_pos[:2], [self.reward_map.shape[1], self.reward_map.shape[0]])
        next_reward=self.reward_map[next_img_pos[1], next_img_pos[0]]
        curremt_img_pos=convert_pos_to_imagepos(self.robot_current_pos[:2], [self.reward_map.shape[1], self.reward_map.shape[0]])
        current_reward=self.reward_map[curremt_img_pos[1], curremt_img_pos[0]]
        reward=reward-next_reward+current_reward

        # success reward
        # if self.global_map[self.goal_image_pos[1], self.goal_image_pos[0]]!=0:
        # if self.observation[self._N_RAYS]*13<self.T_dis:
        if np.linalg.norm(self.robot_pos[:2]-self.goal_pos[:2])<self.T_dis:
            reward=2
            done=True

        info={"robot_pos":self.robot_pos,
                "seen_area": 0,

                }
        info.setdefault("pick", self.pick)

        # print("robot_pos={}, area_reward={}, dis_reward={}".format(self.robot_pos, area_reward, dis_reward))
        return self.observation, reward, done, info

    def render(self):
        pr2_base_conf=(self.robot_pos[0], self.robot_pos[1], self.pr2_yaw)
        update_robot_base(self.robots[0], pr2_base_conf)
        update_rover_base(self.mobile[0], self.rover_pos)
        self.robot_current_pos[:]=self.robot_pos[:]
        # wait_for_duration(0.05)

    def render_steps(self):
        visualize_next_goal(self.statics[-1], self.robot_pos) # next goal
        
        if len(self.pick)>0:
            # when there are movable obstacles to manipulate
            print(self.all_objects)
            print(self.pick)
            # stock region
            visualize_next_goal(self.statics[-3], self.robot_current_pos[:2]-0.5)

            working_limits={"0":[self.robot_current_pos[0]-self.ray_length, self.robot_current_pos[0]+self.ray_length],
                            "1":[self.robot_current_pos[1]-self.ray_length, self.robot_current_pos[1]+self.ray_length]}
            # main_simple_version(self.pick, self, custom_limits=working_limits)
            # for obj in self.pick:
            #     main_simple_version([obj], self)
            print("finish grasp")
        pr2_base_conf=(self.robot_pos[0], self.robot_pos[1], 0)
        is_plan,_=plan_pr2_base_motion(self.robots[0], pr2_base_conf)
        if not is_plan:
            print("reset ")
            self.robot_pos[:2]=self.robot_current_pos[:2]
            print(self.robot_pos[:2])
        update_rover_base(self.mobile[0], self.rover_pos)
        self.robot_current_pos[:]=self.robot_pos[:]

    def close(self):
        disconnect()
        pass

    def random_move(self):
        # generate random number [-0.5,0.5]
        random_offset_x=random.random()-0.5
        random_offset_y=random.random()-0.5
        goal_pos=[self.rover_pos[0,0]+random_offset_x, self.rover_pos[0,1]+random_offset_y, 0.0]
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
