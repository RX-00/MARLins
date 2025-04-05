# MARLins
This project is for training a reinforcement learning (RL) policy that controls multiple robot agents to accomplish a (probably) push-to-place task.

# Installation
The installation process is based on the guide provided by [ROS2swarm](https://github.com/ROS2swarm/ROS2swarm/blob/master/INTALL_GUIDE.md) with some changes for Ubuntu 22.04.5 LTS and ROS 2 Humble (though it hopefully works for other ROS 2).

Please follow the installation guide for ROS2swarm's prerequirements and dependencies up until the `Installation of TurtleBot3 Support` section. To make things easier, the TurtleBot3 packages are included locally in this workspace alongside ROS2swarm rather than a separate one as seen [here](https://github.com/ROS2swarm/ROS2swarm/blob/master/INTALL_GUIDE.md).

## Build
To build the whole project go to the root of your ROS 2 workspace and run

``` bash
colcon build --symlink-install
```
The reason why we use the flag `--symlink-install` is so that colcon creates symbolic links in the install directory that point back to the source files. This means the install space always reflects the current source code, even if a package lacks explicit install rules defined. This allows for convenient development since any change in source is available without always needing to explicitly rebuild. Colcon expects each package to have explicit install targets so if you leave this out some packages might not end up in the install space if they don't have them, leading to runtime issues.

## Environment Configuration
Here's the bash commands for `.bashrc` to properly source this project. Be sure to change the workspace name, `swarm_ws`, to whatever you have.
```bash
source ~/swarm_ws/install/setup.bash
export ROS_DOMAIN_ID=30 #TURTLEBOT3
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/swarm_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models
export TURTLEBOT3_MODEL=waffle_pi
```

## Testing Installation
To test if the TurtleBot3 packages are working you can try running:
``` bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

To test if the ROS2swarm package is properly installed you can try running:
``` bash
bash ~/swarm_ws/src/ROS2swarm/start_simulation.sh 4
```
You can change the number to reflect how many TurtleBots you want to initialize in the Gazebo simulation. This next part will explain what's inside and how to use/modify this script:
Below is an example Markdown file that documents the launch command and explains each argument in detail:

---

# ROS2swarm Launch Command Documentation

This document explains the following command used to launch a ROS 2 simulation with Gazebo using the ROS2swarm package. The command sets up the environment, spawns the robots, and defines the swarm behavior.

```bash
ROS_DOMAIN_ID=42 ros2 launch launch_gazebo create_enviroment.launch.py \
  gazebo_world:=arena_large.world \
  pattern:=dispersion_pattern \
  number_robots:=4 \
  total_robots:=4 \
  log_level:=info \
  robot:=burger \
  sensor_type:=lidar \
  x_start:=0.0 \
  x_dist:=0.0 \
  y_start:=0.0 \
  y_dist:=1.0 \
  driving_swarm:=False \
  logging:=False
```

## Command Breakdown

### 1. Environment Variable

- **`ROS_DOMAIN_ID=42`**  
  Sets the DDS domain ID to 42. In ROS 2, this ensures that only nodes with the same domain communicate, effectively isolating this swarm’s network from other ROS 2 systems.

### 2. Launching the Simulation

- **`ros2 launch launch_gazebo create_enviroment.launch.py`**  
  This part of the command starts the ROS 2 launch system to execute the `create_enviroment.launch.py` file found in the `launch_gazebo` package. This launch file is responsible for starting Gazebo, spawning robots, and configuring the simulation environment.

### 3. Launch Parameters

Each subsequent parameter is passed to the launch file using the syntax `parameter_name:=value`:

- **`gazebo_world:=arena_large.world`**  
  Specifies the world file used by Gazebo. The file `arena_large.world` defines the environment in which the simulation takes place (e.g., arena dimensions, obstacles).

- **`pattern:=dispersion_pattern`**  
  Sets the swarm behavior pattern. Here, the `dispersion_pattern` instructs the robots to spread out from each other once activated.

- **`number_robots:=4`**  
  Indicates the number of robots to instantiate in the simulation (four robots in this example).

- **`total_robots:=4`**  
  Specifies the total expected number of robots. Although similar to `number_robots`, it may be used internally for consistency checks within the launch configuration.

- **`log_level:=info`**  
  Determines the logging verbosity. Setting it to `info` displays informational messages, which is useful for monitoring the simulation status.

- **`robot:=burger`**  
  Selects the type of robot to use. In this case, `burger` refers to the TurtleBot3 Burger model, meaning that the corresponding robot description (URDF) and configurations will be loaded.

- **`sensor_type:=lidar`**  
  Configures the robots to use LiDAR sensors for perception, affecting how sensor data is acquired and processed.

- **`x_start:=0.0` and `y_start:=0.0`**  
  Define the starting coordinates (x and y) for the first robot. With both set to 0.0, the first robot spawns at the origin.

- **`x_dist:=0.0` and `y_dist:=1.0`**  
  Determine the offset between consecutive robots. With an x offset of 0.0 and a y offset of 1.0, each subsequent robot is positioned 1.0 unit further along the y-axis, forming a vertical line.

- **`driving_swarm:=False`**  
  This flag disables a specific coordinated “driving” mode for the swarm. When set to False, the simulation does not engage any additional driving behavior that might alter the default movement pattern.

- **`logging:=False`**  
  Disables extra logging, keeping the console output less verbose. This is useful for performance or clarity during long-running simulations.

## Overall Behavior

When the command is executed, it will:

1. **Set Up Communication:**  
   The `ROS_DOMAIN_ID` ensures that all nodes (robots and simulation components) communicate within domain 42.

2. **Launch the Simulation:**  
   The `create_enviroment.launch.py` launch file starts Gazebo using the `arena_large.world` file and spawns 4 TurtleBot3 Burger robots.

3. **Apply Behavior Configuration:**  
   The swarm behavior is set to `dispersion_pattern`, causing the robots to spread apart once the simulation is running.

4. **Position Robots:**  
   The robots are placed starting at the origin (0.0, 0.0), with subsequent robots offset by 1.0 unit along the y-axis.

5. **Manage Logging and Modes:**  
   Logging is kept at an informational level and additional driving or verbose logging modes are disabled.

This command and its parameters provide a flexible way to control various aspects of the simulation, from the environment and robot placement to the specific swarm behavior being executed.

For additional details and context, please refer to the ROS2swarm codebase documentation citeturn0file0.

--- 

Feel free to adjust or expand on this documentation as needed for your project.

This command simply publishes a message of int8 value 1 to signal the robots to start moving.
``` bash
bash ~/swarm_ws/src/ROS2swarm/start_command.sh
```

## Notes
The repomix-output.xml under the ROS2swarm package is created with [Repomix](https://repomix.com), which is a packing of the whole package into an AI-friendly format for use with LLMs. Instead of using just copilot or an AI IDE like Cursor, I think for now that things like this is more effective. My theory being that robotics tools are more obscure/maybe more difficult so these AI-assisted programming IDEs struggle with them. Meanwhile I find LLMs to be better understand these repos when compacting the whole project into a single file they can read.
