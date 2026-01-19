"""Unit tests for the agent module.

uv run pytest agent_test.py -s
"""

import agent

def test_naive_cyclic_agent_actions():
    agt = agent.NaiveCyclicAgent(num_noops_till_flap=3)
    actions = [agt.next_action(None) for _ in range(10)]
    expected_actions = [
        agent.ActType.IDLE,
        agent.ActType.IDLE,
        agent.ActType.IDLE,
        agent.ActType.FLAP,
        agent.ActType.IDLE,
        agent.ActType.IDLE,
        agent.ActType.IDLE,
        agent.ActType.FLAP,
        agent.ActType.IDLE,
        agent.ActType.IDLE,
    ]
    assert actions == expected_actions