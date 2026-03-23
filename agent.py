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
        self.network_sync_rate = hyperparams["network_sync_rate"]
        self.learning_rate_a = hyperparams["learning_rate_a"]
        self.discount_factor_g = hyperparams["discount_factor_g"]

        self.loss_fn = torch.nn.MSELoss()
        self.policy_network_optimizer = None

    def run(self, is_training=True, render=False):

        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_actions = env.action_space.n
        num_sates = env.observation_space.shape[0]

        # The behavioral policy network. Note DQN is off-policy, and policy_dqn
        # here is the main network on-policy. The target network for off-policy
        # is a different one.
        policy_dqn = DQN(num_sates, num_actions).to(device)

        if is_training:
            replay_buffer = ReplayBuffer(capacity=self.replay_buffer_capacity)
            epsilon = self.epsilon_init

            # Creates the target DQN.
            target_dqn = DQN(num_sates, num_actions).to(device)
            # Copies weights and baises from policy_dqn to target_dqn. They need
            # to be identical at the beginning of training, though as training
            # progresses, target network will diverge to avoid optimizing over
            # a moving target if we were to only have one network.
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Tracks steps taken to determine when to update the target DQN. In
            # the seminal 2015 Nature paper "Human-level control through deep
            # reinforcement learning", this is done every 10,000 steps.
            step_count = 0

            # Policy network optimizer.
            self.policy_network_optimizer = torch.optim.Adam(
                policy_dqn.parameters(), lr=self.learning_rate_a
            )

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

                    step_count += 1

                state = new_state

            rewards_per_episode[episode] = episode_reward

            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)

            # Check if enough experiences have been collected in replay buffer,
            # and if so, optimize policy and target DQNs together.
            if len(replay_buffer) > self.batch_size:
                batch = replay_buffer.sample(self.batch_size)
                self.optimize(batch, policy_dqn, target_dqn)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

    def optimize(
        self, batch: list[Transition], policy_dqn: DQN, target_dqn: DQN
    ) -> None:
        """Optimize the policy and target DQNs together.

        In regular Q-learning, we have:

        ```
        q[state, action] = q[state, action] + alpha * (reward + gamma * max(q[new_state, :]) - q[state, action])
        ```

        For DQN target, by definition of q[state, action], we have:

        ```
        q[state, action] = reward if new_state is terminal else reward + gamma * max(q[new_state, :])
        ```
        """

        # Instead of self._optimize_inefficient, which is more readable, we
        # leverage vecotorized operations to speed up the optimization process.
        states = [transition.state for transition in batch]
        actions = [transition.action for transition in batch]
        rewards = [transition.reward for transition in batch]
        new_states = [transition.new_state for transition in batch]
        terminations = [transition.terminated for transition in batch]
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        new_states = torch.stack(new_states)
        terminations = torch.tensor(terminations).float().to(device)

        # Calculate target q values (expected returns).
        with torch.no_grad():
            target_q = (
                rewards
                + (1 - terminations)
                * self.discount_factor_g
                * target_dqn(new_states).max(dim=1)[0]
            )
            """
            target_dqn(new_states) ==> tensor([[1, 2, 3], [4, 5, 6]])
              .max(dim=1) ==> torch.return_types.max(values=tensor([3, 6]), indices=tensor([3, 0, 0, 1]))
                [0]  ==> tensor([3, 6])
            """

        # Calculate policy q values from current policy.
        current_q = (
            policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        )
        """
        policy_dqn(states) looks like 

        ```
        tensor([[-0.2207,  0.1549],
            [ 0.0648,  0.0782],
            [-0.1931,  0.0639],
            [-0.2361,  0.0767],
            [-0.1340,  0.0462],
            ...
        ```

        `.gather(dim=1, index=actions.unsqueeze(dim=1))` selects the 1st dimension
        according to the actions. actions look like this:

        ```
        tensor([0, 1, 0, 1, 0, ...])
        ```

        and `actions.unsqueeze(dim=1)` looks like this:

        ```
        tensor([[0],
            [1],
            [0],
            [0],
            [1],
            [0],
            ...
        ```

        `policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1))` looks like this:

        ```
        tensor([[ 0.1549],
            [ 0.0782],
            [-0.1931],
            [-0.2361],
            [ 0.0462],
            [ 0.0770],
            [-0.1968],
            [ 0.0885],
            [-0.0743],
            [ 0.0887],
            [ 0.0654],
            ...
        ```

        so we need one last `.squeeze()` to get rid of the extra dimension:

        ```
        tensor([ 0.1549,  0.0782, -0.1931, -0.2361,  0.0462,  0.0770, -0.1968,  0.0885,
                -0.0743,  0.0887,  0.0654,  0.1921, -0.1401,  0.0602,  0.0347,  0.0531,
                -0.1914,  0.1662,  0.0657, -0.1878, -0.0312,  0.0561,  0.0652,  0.0354,
                -0.1866,  0.0109,  0.0642, -0.2321, -0.2385, -0.1706, -0.2369, -0.1900],
            device='mps:0', grad_fn=<SqueezeBackward0>)
        ```
        """

        # Below are as `self._optimize_inefficient`.
        loss = self.loss_fn(current_q, target_q)
        self.policy_network_optimizer.zero_grad()  # clear gradients.
        loss.backward()  # backprop to compute gradients.
        self.policy_network_optimizer.step()  # update *policy* network.

    def _optimize_inefficient(
        self, batch: list[Transition], policy_dqn: DQN, target_dqn: DQN
    ) -> None:
        for state, action, new_state, reward, terminated in batch:
            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target_q = (
                        reward + self.discount_factor_g * target_dqn(new_state).max()
                    )

            # Notice how we decoupled target_dqn from policy_dqn. If we were to
            # use policy_dqn to compute the target, then we would be
            # optimizing over a moving target, which is unstable. By using a
            # separate target_dqn, we can keep the target fixed for a number of
            # steps, and only update it periodically by copying weights from
            # policy_dqn.
            current_q = policy_dqn(state)

            # Compute loss as diff between 2 networks. This is the key ingrident
            # to DQN - we're NOT doing a supervised-learning (there is no y to
            # ground the loss on), but instead we rely on RL to learn the y as
            # the target DQN.
            loss = self.loss_fn(current_q, target_q)

            # Optimize the model. Note the optimizer is strictly for the policy
            # network, NEVER for the target network. In fact, target network is
            # just a old snapshot of the policy network's w eights, but when we
            # compute the loss, the target is the max of q values over all
            # actions of the new_state, making it a decoupled target from the
            # actively evolving q network.
            self.policy_network_optimizer.zero_grad()  # clear gradients.
            loss.backward()  # backprop to compute gradients.
            self.policy_network_optimizer.step()  # update *policy* network.


if __name__ == "__main__":
    agent = Agent("cartpole1")
    agent.run(is_training=True, render=True)
