from stable_baselines3.common.env_checker import check_env
from environment import NAMOENV

env=NAMOENV(init_pos=[-4,-1,0], goal_pos=[4,-1,0])
# test if the environment works
# check_env(env, warn=True)

obs=env.reset()
env.render()
action=[1,0]
n_steps=20
for step in range(n_steps):
    print("Step {}".format(step))
    obs, reward, done, info=env.step(action)
    # print("obs={}, reward={}, done={}".format(obs, reward, done))
    env.render()
    if done:
        print("goal reached!, reward={}".format(reward))
        break