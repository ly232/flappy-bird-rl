"""Driver for simulating Flappy Bird.

uv run python main.py
"""

import gymnasium as gym
import flappy_bird_gymnasium  # Required import to register "FlappyBird-v0"
from gymnasium.wrappers import RecordVideo

if __name__ == '__main__':
    env = gym.make('FlappyBird-v0', render_mode='rgb_array')
    env = RecordVideo(env, video_folder='recordings/', name_prefix='flappy_bird_simulation')

    obs, _ = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
