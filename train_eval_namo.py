from sched import scheduler
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3 import A2C, PPO, SAC
from environment import NAMOENV
from callbacks import TensorboardCallback
from stable_baselines3.common.callbacks import EvalCallback


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


env=NAMOENV(init_pos=[-4,1,0], goal_pos=[4,-1,0], use_gui=False)

env=make_vec_env(lambda: env, n_envs=1)
eval_env=NAMOENV(init_pos=[-4,1,0], goal_pos=[4,1,0], use_gui=False)
eval_env=make_vec_env(lambda: eval_env, n_envs=1)

eval_callback=EvalCallback(eval_env, best_model_save_path="models/best/",
                        eval_freq=5000, deterministic=True, render=False)

model_name="models/ppo_namo_robopos_100000"
model=PPO("MlpPolicy", env, tensorboard_log="./tensorboard/",verbose=1, 
        learning_rate=linear_schedule(1e-3))
# model_name="models/sac_namo_colli_100000"
# model=SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
# model=PPO.load(model_name, env=env)
model.learn(100000,tb_log_name=model_name[7:], callback=eval_callback)
model.save(model_name)
