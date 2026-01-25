"""
uv run python scripts/plot_learning_curve.py
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy


def moving_average(values, window):
    """Smooth values using a moving average."""
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


# Load the monitor data
results = load_results("./logs/")
x, y = ts2xy(results, "timesteps")

# Apply smoothing (window size of 50 episodes)
y_smoothed = moving_average(y, window=50)
# Truncate x to match the smoothed y length
x_smoothed = x[len(x) - len(y_smoothed) :]

plt.figure(figsize=(10, 5))
plt.plot(x, y, alpha=0.2, label="Raw Reward")
plt.plot(x_smoothed, y_smoothed, label="Smoothed Reward (Moving Avg 50)")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("Flappy Bird Learning Curve")
plt.legend()
plt.show()
