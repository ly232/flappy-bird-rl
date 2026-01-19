"""Models an episode of interaction with the environment."""

from gymnasium.wrappers import RecordVideo
from typing import Any

import gymnasium as gym
import flappy_bird_gymnasium  # Required import to register "FlappyBird-v0"

import agent
import uuid

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
        self._episode_id: uuid.UUID = uuid.uuid4()

    def run(self) -> None:
        obs, _ = self._env.reset()
        self._trajectory = [obs]
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = self._agent.next_action(self._episode_id, obs)
            obs, reward, terminated, truncated, info = self._env.step(action)
            self._trajectory.extend([
                action,
                reward,
                obs,
            ])
            self._agent.notify_reward_and_new_state(self._episode_id, reward, obs)

        # Upon termination, signal agent on the final reward.
        self._agent.notify_termination(self._episode_id, self._trajectory)
        print(f'final score: {info.get("final_score", 0)}, final reward: {reward}')

        self._env.close()
