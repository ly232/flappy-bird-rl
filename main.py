"""Driver for simulating Flappy Bird.

uv run python main.py
"""

import episode
import agent


_NUM_EPISODES = 10


if __name__ == '__main__':
    agt = agent.NaiveCyclicAgent()
    for _ in range(_NUM_EPISODES):
        ep = episode.Episode(agt)
        ep.run()