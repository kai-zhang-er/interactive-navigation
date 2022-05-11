from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3 import A2C, PPO, SAC
from environment import NAMOENV


env=NAMOENV(init_pos=[-4,-1,0], goal_pos=[4,-1,0], use_gui=True)

model=PPO.load("models/ppo_navigation10000")

num_episodes=5
max_steps=1000
for i in range(num_episodes):
    obs=env.reset()
    env.render()
    print("{}".format(env.robot_pos))
    for j in range(max_steps):
        action, _=model.predict(obs)
        obs, rewards, dones, info=env.step(action)
        # print("pos: {}, reward={}, done={}".format(info["robot_pos"], rewards, dones))
        env.render()
        if dones:
            print("goal reached!, steps={},reward={}".format(j, rewards))
            break
    