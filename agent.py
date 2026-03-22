import flappy_bird_gymnasium
import gymnasium
import itertools
import random
import torch
import yaml

from dqn import DQN
from replay_buffer import ReplayBuffer, Transition

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"


class Agent:

    def __init__(self, app_name="cartpole1"):
        with open("hyperparameters.yaml", "r") as f:
            self.config = yaml.safe_load(f)
            hyperparams = self.config[app_name]
        self.replay_buffer_capacity = hyperparams["replay_buffer_capacity"]
        self.batch_size = hyperparams["batch_size"]
        self.epsilon_init = hyperparams["epsilon_init"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.epsilon_min = hyperparams["epsilon_min"]

    def run(self, is_training=True, render=False):

        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_actions = env.action_space.n
        num_sates = env.observation_space.shape[0]

        policy_dqn = DQN(num_sates, num_actions).to(device)

        if is_training:
            replay_buffer = ReplayBuffer(capacity=self.replay_buffer_capacity)
            epsilon = self.epsilon_init

        rewards_per_episode = {}
        epsilon_history = []

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0

            while not terminated:

                # Epsilon-greedy action selection.
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    action = (
                        policy_dqn(torch.tensor(state, device=device).unsqueeze(dim=0))
                        .squeeze(dim=0)
                        .argmax()
                    )

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
                new_state, reward, terminated, _, info = env.step(action.item())

                # Convert to tensors.
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                episode_reward += reward

                if is_training:
                    transition = Transition(
                        state, action, reward, new_state, terminated
                    )
                    replay_buffer.append(transition)

                state = new_state

            rewards_per_episode[episode] = episode_reward

            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)


if __name__ == "__main__":
    agent = Agent("cartpole1")
    agent.run(is_training=True, render=True)
