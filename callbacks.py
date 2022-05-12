from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnRewardThreshold


class TensorboardCallback(BaseCallback):
    """
    custom callback for plotting more values on tensorboard
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record("train/training_rewards", self.parent.best_mean_reward)
        return super()._on_step()