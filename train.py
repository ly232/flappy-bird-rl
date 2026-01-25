"""Initiates training process.

uv run python train.py
"""

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback


import env_factory
import gymnasium as gym

_NUM_TRAIN_STEPS = 100000


def train_dqn(env: gym.Env):
    """Trains a DQN agent in the given environment."""
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        tensorboard_log="./logs/tensorboard/",
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./ckpt/",
        name_prefix="dqn_flappy_bird_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    model.learn(
        total_timesteps=_NUM_TRAIN_STEPS,
        callback=checkpoint_callback,
        log_interval=10,
        progress_bar=True,
    )
    model.save("./model/dqn_flappy_bird")


if __name__ == "__main__":
    factory = env_factory.EnvFactory()
    train_dqn(factory.create_training_env())
