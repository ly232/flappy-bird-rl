import flappy_bird_gymnasium
import gymnasium
import itertools
import torch

from dqn import DQN
from replay_buffer import ReplayBuffer, Transition

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"


class Agent:

    def run(self, is_training=True, render=False):

        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_actions = env.action_space.n
        num_sates = env.observation_space.shape[0]

        policy_dqn = DQN(num_sates, num_actions).to(device)

        if is_training:
            replay_buffer = ReplayBuffer(capacity=10000)

        rewards_per_episode = {}

        for episode in itertools.count():
            state, _ = env.reset()
            terminated = False
            episode_reward = 0

            while not terminated:
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                # Example obs: [0.9861111111111112, 0.234375, 0.4296875, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.4609375, -0.8, 0.4666666666666667]
                # the last pipe's horizontal position
                # the last top pipe's vertical position
                # the last bottom pipe's vertical position
                # the next pipe's horizontal position
                # the next top pipe's vertical position
                # the next bottom pipe's vertical position
                # the next next pipe's horizontal position
                # the next next top pipe's vertical position
                # the next next bottom pipe's vertical position
                # player's vertical position
                # player's vertical velocity
                # player's rotation
                new_state, reward, terminated, _, info = env.step(action)
                episode_reward += reward

                if is_training:
                    transition = Transition(
                        state, action, reward, new_state, terminated
                    )
                    replay_buffer.append(transition)

                state = new_state

            rewards_per_episode[episode] = episode_reward
