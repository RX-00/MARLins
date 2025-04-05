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
You can change the number to reflect how many TurtleBots you want to initialize in the Gazebo simulation.

The other scripts do...
