"""Models an episode of interaction with the environment."""

from gymnasium.wrappers import RecordVideo
from typing import Any

import gymnasium as gym
import flappy_bird_gymnasium  # Required import to register "FlappyBird-v0"

import agent

class Episode():
    """Models a a single episode of Flappy Bird."""

    def __init__(self, agt: agent.Agent):
        """Initializes the episode.
        
        Args:
            agt: The agent to use for action selection.
        """
        self._env = gym.make('FlappyBird-v0', render_mode='rgb_array')
        self._env = RecordVideo(
            self._env, video_folder='recordings/', 
            name_prefix=agt.__class__.__name__,
        )
        self._agent = agt
        self._trajectory: list[Any] = []

    def run(self):
        obs, _ = self._env.reset()
        self._trajectory = [obs]
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = self._agent.next_action(obs)
            obs, reward, terminated, truncated, info = self._env.step(action)
            self._trajectory.extend([
                action,
                reward,
                obs,
            ])

        # Upon termination, signal agent on the final reward.
        self._agent.notify_termination(reward, self._trajectory)

        self._env.close()
