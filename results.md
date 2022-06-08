test_num=100, max_step=1000

- ppo_namo_ang_nosub_100000_log: PPO + angular_linear_speed+no_negative_reward+100000_train_steps+log_distance_reward
sucess episodes: 100, success rate=1.0, average steps=232.73976726023272, average_picks=1.81999818000182

- ppo_namo_ang_nosub_100000_log_lesspick: + -1_reward_to_less_pick
sucess episodes: 92, success rate=0.92, average steps=402.8478260869565

- ppo_namo_ang_nosub_100000_log_ray2_lesspick: ray_length=2
sucess episodes: 87, success rate=0.87, average steps=381.4593316559406

- ppo_namo_ang_nosub_100000_log_ray2
sucess episodes: 95, success rate=0.95, average steps=342.66279719705557, average_picks=3.4526279446021633

- ppo_namo_da_100000_anglediff: set observation= ray+distance+ angle_difference
sucess episodes: 82, success rate=0.82, average steps=491.99940000073167, average_picks=0.0

- ppo_namo_allobs_100000: all data are from observation, only success reward + bad next goal penalty, 
sucess episodes: 100, success rate=1.0, average steps=35.45996454003546, average_picks=0.659999340

- PPO_preTrained/mynamo/PPO_mynamo_0_rotatefree.pth  only positive linear speed
sucess episodes: 100, success rate=1.0, average steps=8.88999111000889, average_picks=0.54999945000055