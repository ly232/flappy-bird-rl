"""Evaluates the trained model.

uv run python eval.py --agent_type=dqn
"""

import argparse
import agent
import env_factory
import gymnasium as gym


def eval_loop(agent: agent.Agent, env: gym.Env) -> None:
    """Runs a single episode to loop until termination."""
    obs, _ = env.reset()
    trajectory = [obs]
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.next_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.extend(
            [
                action,
                reward,
                obs,
            ]
        )
        agent.notify_reward_and_new_state(reward, obs)

    agent.notify_termination(trajectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "agent_type",
        type=str,
        choices=["naive_cyclic", "dqn"],
        help="Type of agent to evaluate.",
    )
    args = parser.parse_args()
    agent_type = args.agent_type

    env = env_factory.EnvFactory().create_evaluation_env(agent_type)
    match agent_type:
        case "dqn":
            agent = agent.DQNAgent(env)
        case "naive_cyclic":
            agent = agent.NaiveCyclicAgent()
        case _:
            raise ValueError(f"Unknown agent type: {agent_type}")
    eval_loop(agent, env)
    env.close()
