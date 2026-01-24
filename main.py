"""Driver for simulating Flappy Bird.

uv run python main.py
"""

import agent
import episode
import tqdm


_NUM_EPISODES = 1


def run(agt: agent.Agent, num_episodes: int = _NUM_EPISODES) -> None:
    """Runs multiple episodes sequentially using the given agent."""
    for _ in tqdm.tqdm(range(num_episodes), desc="Episodes"):
        episode.Episode(agt).run()


if __name__ == "__main__":
    # print('====== Running NaiveCyclicAgent ======')
    # run(agent.NaiveCyclicAgent())
    # print("====== Running MonteCarloTabularAgent ======")
    # mc_agent = agent.MonteCarloTabularAgent(gamma=0.9)
    # run(mc_agent)
    # mc_agent.plot_total_rewards()
    print("====== Running DQNAgent ======")
    dqn_agent = agent.DQNAgent()
    run(dqn_agent)
    dqn_agent.plot_total_rewards()
