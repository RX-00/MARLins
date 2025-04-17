import os
os.environ["PYGAME_DETECT_AVX2"] = "1"
import numpy as np
import gymnasium as gym
import gym_kilobots

env = gym.make('Kilobots-QuadAssembly-v0', render_mode='human')

obs, info = env.reset()

for t in range(300):
    env.render()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break

env.close()  # Ensure the environment is properly closed
