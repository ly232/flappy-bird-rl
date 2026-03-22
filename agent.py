import flappy_bird_gymnasium
import gymnasium

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
# env = gymnasium.make("CartPole-v1", render_mode="human")

obs, _ = env.reset()
while True:
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
    obs, reward, terminated, _, info = env.step(action)

    # Checking if the player is still alive
    if terminated:
        break

env.close()
