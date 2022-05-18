from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3 import A2C, PPO, SAC
from environment import NAMOENV
from utils.pybullet_tools.utils import wait_for_duration, wait_for_user


# env=NAMOENV(init_pos=[-4,1,0], goal_pos=[4,1,0], use_gui=False)
env=NAMOENV(init_pos=[-4,1,0], goal_pos=[4,1,0], use_gui=True)

model=PPO.load("models/ppo_namo_ang_nosub_100000_log")

num_episodes=5
finish_steps=[]
success_rate=0
max_steps=1000
num_picks=0
for i in range(num_episodes):
    obs=env.reset()
    env.render()
    print("{}".format(env.robot_pos))
    # wait_for_user()
    for j in range(max_steps):
        action, _=model.predict(obs)
        obs, rewards, dones, info=env.step(action)
        if info["pick"]>-1:
            num_picks+=1
        # print("pos: {}, reward={}, done={}".format(info["robot_pos"], rewards, dones))
        env.render_steps()
        # wait_for_duration(0.01)
        if dones:
            finish_steps.append(j)
            print("goal reached!, steps={},reward={}, picks={}".format(j, rewards, num_picks))
            break
    
success_rate=len(finish_steps)/num_episodes
average_finish_steps=sum(finish_steps)/(len(finish_steps)+0.0001)
print("sucess episodes: {}, success rate={}, average steps={}".format(len(finish_steps),success_rate, average_finish_steps))

    