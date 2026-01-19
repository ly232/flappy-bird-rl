"""Module of all agents implementing different RL algorithms."""

from flappy_bird_gymnasium.envs.flappy_bird_env import Actions
from jaxtyping import Float
from typing import override, Any

import abc
import collections
import itertools

import numpy as np
import uuid


# Observation assumes Lidar mode, which returns 180 distance readings from the
# bird to the nearest obstacle in each degree direction.
ObsType = Float[np.ndarray, "180"]
ActType = Actions


class Agent(abc.ABC):
    """Abstract base class for all agents."""

    @abc.abstractmethod
    def next_action(
        self, episode_id: uuid.UUID, observation: ObsType) -> Actions:
        """Selects an action based on the given observation.
        
        Args:
            episode_id: The unique identifier for the current episode.
            observation: The current observation from the environment.
        
        Returns:
            The action selected by the agent.
        """
        pass

    @abc.abstractmethod
    def notify_reward_and_new_state(
        self, episode_id: uuid.UUID, reward: float, new_observation: ObsType):
        """Notifies the agent of a received reward and new observation.

        The environment is responsible for always calling this method after
        calling `step(action)` for an action produced by `next_action(observation)`. 
        
        Args:
            episode_id: The unique identifier for the current episode.
            reward: The reward received after taking an action.
            new_observation: The new observation from the environment.
        """
        pass

    @abc.abstractmethod
    def notify_termination(self, episode_id: uuid.UUID, trajectory: list[Any]):
        """Notifies the agent of episode termination.
        
        Args:
            episode_id: The unique identifier for the current episode.
            trajectory: The full trajectory of the episode.
        """
        pass


class NaiveCyclicAgent(Agent):
    """An naive agent that cycles through no-op and flap actions."""

    def __init__(self, num_noops_till_flap: int = 15):
        """Initializes the agent.
        
        Args:
            num_noops_till_flap: Number of no-op actions before a flap action.
        """
        self._action_cycles: dict[uuid.UUID, itertools.cycle] = collections.defaultdict(
            lambda: itertools.cycle(
                [ActType.IDLE] * num_noops_till_flap + [ActType.FLAP]))

    @override
    def next_action(
        self, episode_id: uuid.UUID, observation: ObsType) -> ActType:
        return next(self._action_cycles[episode_id])
    
    @override
    def notify_reward_and_new_state(
        self, episode_id: uuid.UUID, reward: float, new_observation: ObsType):
        pass
    
    @override
    def notify_termination(
        self, episode_id: uuid.UUID, trajectory: list[Any]):
        pass


class MonteCarloTabularAgent(Agent):
    """An agent that uses Monte Carlo every-visit tabular method to learn a policy."""

    def __init__(self, gamma: float = 0.9):
        self._q: dict[tuple[ObsType, ActType], float] = collections.defaultdict(float)
        self._n: dict[tuple[ObsType, ActType], int] = collections.Counter()
        self._gamma = gamma
        self._episode_count = 1

    def _serialize_observation(self, obs: ObsType) -> str:
        """Serializes the raw observations.
        
        Note that to simplify state space, we only keep the middle 60 readings,
        corresponding to +/- 30 degrees in front of the bird).
        """
        return tuple(np.round(obs[30:150:1], 2))

    @override
    def next_action(
        self, episode_id: uuid.UUID, observation: ObsType) -> ActType:
        observation = self._serialize_observation(observation)
        # Greedy exploitation.
        best_action, best_value = None, float('-inf')
        for action in ActType:
            q_value = self._q[(observation, action)]
            if q_value > best_value:
                best_value = q_value
                best_action = action
        # Epsilon-greedy exploration.
        # eps = 1 / self._episode_count
        eps = max(0.1, 1.0 / np.sqrt(self._episode_count + 1))
        probabilities = {
            action: (
                eps / len(ActType) + (1 - eps) if action == best_action 
                else (eps / len(ActType))
            )
            for action in ActType
        }
        actions = list(probabilities.keys())
        weights = list(probabilities.values())
        chosen_action = np.random.choice(actions, p=weights)
        return chosen_action

    @override
    def notify_reward_and_new_state(
        self, episode_id: uuid.UUID, reward: float, new_observation: ObsType):
        # No-op because MC only updates at episode termination.
        pass

    @override
    def notify_termination(self, episode_id: uuid.UUID, trajectory: list[Any]):
        self._episode_count += 1
        # Construct visits counter.
        gain = 0.0
        states = trajectory[0::3]
        actions = trajectory[1::3]
        rewards = trajectory[2::3]
        for s, a, r in reversed(list(zip(states, actions, rewards))):
            gain = r + self._gamma * gain
            key = (self._serialize_observation(s), a)
            self._n[key] += 1
            self._q[key] += \
                1 / self._n[key] * (gain - self._q[key])
