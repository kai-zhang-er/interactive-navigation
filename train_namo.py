from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3 import A2C, PPO, SAC
from environment import NAMOENV


env=NAMOENV(init_pos=[-4,-1,0], goal_pos=[4,-1,0], use_gui=False)

env=make_vec_env(lambda: env, n_envs=1)
model_name="models/ppo_navigation"
model=PPO("MlpPolicy", env, tensorboard_log="./tensorboard/",verbose=1)
model.learn(20000)
model.save(model_name)
