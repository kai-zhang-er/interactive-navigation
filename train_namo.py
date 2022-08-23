from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import A2C, PPO, SAC
from environment_forward_door import NAMOENV
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback

def linear_schedule(initial_value: float)

seed=11
set_random_seed(seed)
env=NAMOENV(init_pos=[-4,1,0], goal_pos=[4,-1,0], use_gui=False)
eval_env=NAMOENV(init_pos=[-4,1,0], goal_pos=[4,1,0], use_gui=False)
env=make_vec_env(lambda: env, n_envs=1)
eval_env=make_vec_env(lambda: eval_env, n_envs=1)

env.seed(seed)
total_timesteps=50000

model_name="models/ppo_env2_forward_switchdoor_{}_{}".format(seed, total_timesteps)
model=RecurrentPPO("MlpLstmPolicy", env, verbose=1, learning_rate=1e-3, tensorboard_log="./tensorboard/")
# model.learn

save_callback=CheckpointCallback(save_freq=10000, save_path="models/")


# model=PPO("MlpPolicy", env, tensorboard_log="./tensorboard/",verbose=1, learning_rate=1e-3)
# finetune
# model=PPO.load("models/ppo_namo_da_100000_test", env=env)
model.learn(total_timesteps,tb_log_name=model_name[7:], callback=save_callback)
model.save(model_name)