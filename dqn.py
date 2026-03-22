import torch
import torch.nn.functional as F

from jaxtyping import Float


class DQN(torch.nn.Module):
    """A simple feedforward neural network for Deep Q-Network (DQN) algorithm."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """Initializes a simple feedforward neural network for DQN.

        Args:
            state_dim: The dimension of the state space.
            action_dim: The dimension of the action space.
            hidden_dim: The number of hidden units in the hidden layer.

        Attributes:
            fc1: The first fully connected layer.
            fc2: The second fully connected layer.
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(
        self, x: Float[torch.Tensor, "batch_size state_dim"]
    ) -> Float[torch.Tensor, "batch_size action_dim"]:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    # Example usage:
    state_dim = 12  # Example state dimension for Flappy Bird
    action_dim = 2  # Example action dimension (e.g., flap or do nothing)
    dqn = DQN(state_dim, action_dim)

    # Example input: batch of states
    batch_size = 4
    example_states = torch.randn(batch_size, state_dim)
    q_values = dqn(example_states)
    print(q_values)
