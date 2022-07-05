import numpy as np

from environment import NAMOENV

total_test_episodes=100

success_steps=[]

env = NAMOENV(init_pos=[-4,0,0], goal_pos=[4,-1,0], use_gui=False)
for ep in range(1, total_test_episodes+1):
    ep_reward = 0
    state = env.reset()

    success, base_path=env.only_navigate()
    if success:
        success_steps.append(len(base_path)*0.03)
    
success_rate=len(success_steps)/(total_test_episodes+0.00001)
average_length=sum(success_steps)/(len(success_steps)+0.00001)

print("success rate={}, average_length={}".format(success_rate, average_length))
