from random import random
from stable_baselines3.common.env_checker import check_env
from environment_explore import NAMOENV
from utils.pybullet_tools.utils import wait_for_user

env=NAMOENV()
# test if the environment works
# check_env(env, warn=True)

obs=env.reset()
env.render()
action=[0.8,0.3]
n_steps=100
for step in range(n_steps):
    print("Step {}".format(step))
    action=[random(),random()]
    obs, reward, done, info=env.step(action)
    # print("obs={}, reward={}, done={}".format(obs, reward, done))
    env.render_steps()
    # wait_for_user()
    if done:
        print("goal reached!, reward={}".format(reward))
        break

