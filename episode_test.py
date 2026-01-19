"""Unit tests for episode.py

uv run pytest episode_test.py -s

This test is vibe-coded :)
"""

import episode

def test_episode(monkeypatch):

    class DummyEnv:
        def __init__(self):
            self.closed = False
            self._step_count = 0

        def reset(self):
            return "obs0", {}

        def step(self, action):
            self._step_count += 1
            if self._step_count == 1:
                return "obs1", 1.0, False, False, {}
            return "obs2", 2.0, True, False, {}

        def close(self):
            self.closed = True

    def fake_make(name, render_mode=None):
        assert name == 'FlappyBird-v0'
        return DummyEnv()

    # Patch gym.make used inside episode.Episode
    monkeypatch.setattr(episode.gym, 'make', fake_make)

    # Patch the RecordVideo symbol imported in episode (returns env unchanged)
    monkeypatch.setattr(episode, 'RecordVideo', lambda env, **kwargs: env)

    class MockAgent:
        def __init__(self):
            self.actions = []
            self.notified = False
            self.last_reward = None
            self.last_traj = None

        def next_action(self, obs):
            self.actions.append(obs)
            return 'action'

        def notify_termination(self, reward, trajectory):
            self.notified = True
            self.last_reward = reward
            self.last_traj = trajectory

    agt = MockAgent()
    ep = episode.Episode(agt)
    ep.run()

    assert agt.notified is True
    assert agt.last_reward == 2.0
    assert agt.last_traj == [
        'obs0', 'action', 1.0,
        'obs1', 'action', 2.0,
        'obs2'
    ]
