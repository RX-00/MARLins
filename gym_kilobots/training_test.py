import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import gym_kilobots

# Define four objects in a centered square
square_objects = [
    ((-0.25,  0.25), 0.0),   # top-left
    (( 0.25,  0.25), 0.0),   # top-right
    (( 0.25, -0.25), 0.0),   # bottom-right
    ((-0.25, -0.25), 0.0),   # bottom-left
]

# Define 1 object in the center of the environment
single_center_object = [
    ((-0.3, 0.3), 0.0)        # top left,
]

# Choose a fixed light position
light_pos = (-0.5, 0.5)

# Choose four explicit kilobot positions
kb_positions = [
    (-0.55, 0.55),
    (-0.50, 0.50),
    (-0.60, 0.60),
    (-0.60, 0.50),
]

# NOTE: if you don't do this, the environment will randomly place the objects
env = gym.make(
    'Kilobots-QuadAssembly-v0',
    render_mode='human',
    num_kilobots=len(kb_positions),
    object_config=single_center_object,
    light_position=light_pos,
    kilobot_positions=kb_positions
)

# Check if the environment is valid
check_env(env,warn=True)

from stable_baselines3.common.callbacks import BaseCallback

class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            # reset may return (obs, info) or just obs
            reset_res = self.env.reset()
            if isinstance(reset_res, tuple):
                obs, _ = reset_res
            else:
                obs = reset_res

            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                step_res = self.env.step(action)

                # unpack step for Gym 0.26+ or SB3 VecEnv API
                if len(step_res) == 5:
                    obs, reward, terminated, truncated, info = step_res
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_res

                self.env.render()
        return True


# Train the agent
#model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=1_000_000,
#            callback=RenderCallback(env, render_freq=10_000))
#model.learn(total_timesteps=1_000_000)
#model.save("swarm_ppo")

# Test the policy
model = PPO.load("swarm_ppo")

def controller(x):
      # Extract only the action from the tuple returned by model.predict
      action, _ = model.predict(x)
      return action

# Create a separate evaluation environment
eval_gym = gym.make(
    'Kilobots-QuadAssembly-v0',
    render_mode='human',
    num_kilobots=len(kb_positions),
    object_config=single_center_object,
    light_position=light_pos,
    kilobot_positions=kb_positions
)

running = True
obs, info = eval_gym.reset()

while running:
      eval_gym.render()

      x = obs
      action = controller(x)

      obs, reward, terminated, truncated, info = eval_gym.step(action)

      # Print the orientation of the objects using the unwrapped environment
      print(eval_gym.unwrapped.get_objects_status())

      # End episode
      if terminated or truncated:
            break

eval_gym.close()  # Ensure the environment is properly closed