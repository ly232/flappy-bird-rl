"""Module of all agents implementing different RL algorithms."""

from flappy_bird_gymnasium.envs.flappy_bird_env import Actions
from jaxtyping import Float
from typing import override, Any

import abc
import itertools

import numpy as np


# Observation assumes Lidar mode, which returns 180 distance readings from the
# bird to the nearest obstacle in each degree direction.
ObsType = Float[np.ndarray, "180"]
ActType = Actions


class Agent(abc.ABC):
    """Abstract base class for all agents."""

    @abc.abstractmethod
    def next_action(self, observation: ObsType) -> Actions:
        """Selects an action based on the given observation.
        
        Args:
            observation: The current observation from the environment.
        
        Returns:
            The action selected by the agent.
        """
        pass

    @abc.abstractmethod
    def notify_reward_and_new_state(
        self, reward: float, new_observation: ObsType):
        """Notifies the agent of a received reward and new observation.

        The environment is responsible for always calling this method after
        calling `step(action)` for an action produced by `next_action(observation)`. 
        
        Args:
            reward: The reward received after taking an action.
            new_observation: The new observation from the environment.
        """
        pass

    @abc.abstractmethod
    def notify_termination(self, final_reward: float, trajectory: list[Any]):
        """Notifies the agent of episode termination.
        
        Args:
            final_reward: The final reward received upon termination.
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
        self._action_cycle = itertools.cycle(
            [ActType.IDLE] * num_noops_till_flap + [ActType.FLAP]
        )

    @override
    def next_action(self, observation: ObsType) -> ActType:
        return next(self._action_cycle)
    
    @override
    def notify_reward_and_new_state(
        self, reward: float, new_observation: ObsType):
        pass
    
    @override
    def notify_termination(self, final_reward: float, trajectory: list[Any]):
        final_stats = f'''
        Episode terminated with final reward: {final_reward}
        Total steps taken: {len(trajectory) // 3}
        '''
        print(final_stats)
        # for elem in trajectory:
        #     if isinstance(elem, ObsType):
        #         print(f'Observation: {len(elem)} lidar readings.')
        #     elif isinstance(elem, ActType):
        #         print(f'Action taken: {elem}.')
        #     else:
        #         print(f'Reward received: {elem}.')
