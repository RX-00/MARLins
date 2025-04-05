# MARLins
This project is for training a reinforcement learning (RL) policy that controls multiple robot agents to accomplish a (probably) push-to-place task.

# Installation
The installation process is based on the guide provided by [ROS2swarm](https://github.com/ROS2swarm/ROS2swarm/blob/master/INTALL_GUIDE.md) with some changes for Ubuntu 22.04.5 LTS and ROS 2 Humble (though it hopefully works for other ROS 2).

Please follow the installation guide for ROS2swarm up until the `Installation of TurtleBot3 Support`

``` 
source ~/swarm_ws/install/setup.bash
export ROS_DOMAIN_ID=30 #TURTLEBOT3
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/swarm_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models
export TURTLEBOT3_MODEL=waffle_pi
```
