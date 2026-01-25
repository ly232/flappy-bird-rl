"""Factory to create Flappy Bird environments."""

from gymnasium.wrappers import RecordVideo
from typing import Any

import os
import gymnasium as gym
import flappy_bird_gymnasium  # Required import to register "FlappyBird-v0"

from functools import cache
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


class EnvFactory:
    """Factory to create Flappy Bird environments."""

    def __init__(self):
        self.training_env: gym.Env | None = None
        self.evaluation_env: gym.Env | None = None

    @cache
    def create_training_env(self, n_parallel: int = 8) -> gym.Env:
        """Creates a vectorized environment for parallel training.

        Each worker writes a Monitor CSV file into `logs/` so episode
        rewards can be aggregated after training for plotting.
        """

        os.makedirs("logs", exist_ok=True)

        def make_env(rank: int):
            def _init():
                env = gym.make("FlappyBird-v0")
                monitor_path = os.path.join("logs", f"monitor_{rank}.csv")
                env = Monitor(env, filename=monitor_path)
                return env

            return _init

        return SubprocVecEnv([make_env(i) for i in range(n_parallel)])

    @cache
    def create_evaluation_env(self, name_prefix: str = "eval") -> gym.Env:
        """Creates an environment for evaluation with video recording.

        Args:
            name_prefix: An optional identifier for the evaluation client.
        """
        env = gym.make("FlappyBird-v0", render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder="eval_recordings/",
            name_prefix=name_prefix,
        )
        return env
