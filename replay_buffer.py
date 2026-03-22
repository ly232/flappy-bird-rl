from collections import deque, namedtuple

import random

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """A simple FIFO replay buffer with fixed size."""

    def __init__(self, capacity):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def append(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
