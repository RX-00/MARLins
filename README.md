# MARLins with gym_kilobot

A custom suite of Gymnasium environment, `gym_kilobots` for simulating “kilobots,” a type of swarm robot, in a 2D world with Box2D. This package supports multiple variations of kilobot control (direct control, phototaxis, etc.) and includes environment configurations for tasks like “QuadAssembly” or “Yaml-based” environment definitions.

This is an updated version of: https://github.com/gregorgebhardt/gym-kilobots

NOTE: This was originally going to use ROS 2 and Gazebo, which isn't ideal for RL training, hence why we updated this old gym environment. The original ROS 2 workspace can be found in swarm_ws.

## 1. Features

- **Box2D-based** physics simulation with kilobot models.
- **Multiple environment classes** for standard tasks:
  - `QuadAssemblyKilobotsEnv`
  - `YamlKilobotsEnv` (load environment setup from YAML)
  - `DirectControlKilobotsEnv`  
- **Rendering** with a custom PyGame-based viewer.
- **Gymnasium**-compatible, so you can use standard RL libraries (PPO, SAC, etc.) with these environments.

---

## 2. Environment Setup

We provide an `environment.yml` file to handle most dependencies via conda. Since Gymnasium 1.1.1 is not currently on conda, you’ll need to `pip install` it after creating the conda environment.

### 2.1 Create and Activate the Conda Environment

```bash
# From the project root (where environment.yml is located)
conda env create -f environment.yml -c conda-forge

# Activate the newly created environment
conda activate swarm
```

### 2.2 Install Gymnasium 1.1.1 and stable_baselines3 via pip

Inside your activated conda environment, run:

```bash
pip install gymnasium==1.1.1
pip install stable_baslines3
```

This ensures the version matches what’s not yet available on conda. Verify:

```bash
python -c "import gymnasium; print(gymnasium.__version__)"
# should output 1.1.1
```

---

## 3. Installing **gym_kilobots** Gymnasium Environment

From the source repo folder, run:

```bash
pip install -e .
```

The `-e` (editable) flag means any local code changes are reflected without re-installation.

---

## 4. Usage

### 4.1 Import and Make an Environment

```python
import gymnasium as gym
import gym_kilobots  # triggers environment registration

env = gym.make("Kilobots-QuadAssembly-v0", render_mode="human")
obs, info = env.reset()
for _ in range(500):
    action = env.action_space.sample() 
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
env.close()
```

- Use `render_mode=None` to run without GUI (no PyGame window).

### 4.2 Example: Running the Built-In `test.py`

There is a `test.py` script in `gym_kilobots` that demonstrates usage of one environment:

```bash
python gym_kilobots/test.py
```

This script will create the “QuadAssembly” environment, reset, and run for a few hundred steps. If everything is correct, a PyGame window appears with kilobots moving around (assuming `render_mode='human'` in the code).

---

## 5. Notes and Troubleshooting

1. **Mixing Conda & Pip**  
   - We install Gymnasium 1.1.0 via pip, and everything else via `environment.yml` in conda. This is generally safe as long as no version conflicts arise, but keep an eye out for warnings.

2. Please refer to the other branches for further progress on the manipulation policy.
