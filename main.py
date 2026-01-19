"""Driver for simulating Flappy Bird.

uv run python main.py
"""

from concurrent.futures import ThreadPoolExecutor

import episode
import agent


_NUM_EPISODES = 10


if __name__ == '__main__':
    agt = agent.NaiveCyclicAgent()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(episode.Episode(agt).run) 
            for _ in range(_NUM_EPISODES)
        ]
    for future in futures:
        future.result()