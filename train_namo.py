from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3 import A2C, PPO, SAC
from environment import NAMOENV
from callbacks import TensorboardCallback
from stable_baselines3.common.callbacks import EvalCallback



env=NAMOENV(init_pos=[-4,1,0], goal_pos=[4,-1,0], use_gui=False)

eval_env=NAMOENV(init_pos=[-4,1,0], goal_pos=[4,1,0], use_gui=False)
env=make_vec_env(lambda: env, n_envs=1)
eval_env=make_vec_env(lambda: eval_env, n_envs=1)

eval_callback=EvalCallback(eval_env, best_model_save_path="models/best/", 
                            log_path="logs/",eval_freq=5000, 
                            deterministic=True, render=False)
model_name="models/ppo_nav_ang_10000_nosub_test"
model=PPO("MlpPolicy", env, tensorboard_log="./tensorboard/",verbose=1)
model.learn(10000,tb_log_name=model_name[7:], callback=eval_callback)
model.save(model_name)
