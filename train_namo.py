from statistics import mode
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3 import A2C, PPO, SAC
from environment import NAMOENV
from callbacks import TensorboardCallback


env=NAMOENV(init_pos=[-4,1,0], goal_pos=[4,-1,0], use_gui=False)

env=make_vec_env(lambda: env, n_envs=1)

model_name="models/ppo_namo_allobs_100000"
model=PPO("MlpPolicy", env, tensorboard_log="./tensorboard/",verbose=1, learning_rate=1e-3)
# model=SAC("MlpPolicy", env, tensorboard_log="./tensorboard/",verbose=1, learning_rate=1e-3)
# model=PPO.load(model_name, env=env)
model.learn(100000,tb_log_name=model_name[7:])
model.save(model_name)
