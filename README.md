# Requirements

## Installing ROS Rolling on ubuntu 24.04

Set locale
```bash
#   if you don't have these tools:
sudo apt-get install sudo git wget gnupg curl

locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # verify settings
```

Install ROS2
```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
```

```bash
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt update
sudo apt upgrade

sudo apt install ros-dev-tools
sudo apt-get install ros-rolling-ros-base
sudo apt-get install ros-rolling-ros-gz
```


Install gazebo
```bash

# Gazebo
sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-prerelease $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-prerelease.list > /dev/null
sudo apt-get update
sudo apt-get install gz-jetty  ros-rolling-ros-gz-bridge  ros-rolling-ros-gz-sim
```

Install opencv
```bash
# Download to a known path
mkdir  ~/src/
cd ~/src/
git clone https://github.com/opencv/opencv
git -C opencv checkout 4.x

git clone https://github.com/opencv/opencv_contrib
git -C opencv_contrib checkout 4.x

mkdir build
cd build

cmake ../opencv
# Any other needed changes can be done with the following command
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv
# make
make -j$(nproc)

# Install
# Usually installs to  /usr/local
sudo make install
```


Install dependencies (messages, compiling tools, cv_bridge and rqt plugins)
```bash
sudo apt-get install python3-rosinstall-generator \
    python3-transforms3d \
    ros-rolling-tf-transformations \
    ros-rolling-ament-lint-auto \
    ros-rolling-ament-cmake \
    ros-rolling-cv-bridge \
    ros-rolling-rqt \
    ros-rolling-rqt-common-plugins
```

##  Install OpenCV:

Install openCV from source (version 4.x):
<https://docs.opencv.org/4.12.0/d0/d3d/tutorial_general_install.html>

`/usr/local` address wil be used, if reuired, change ir at cmake config:
```bash
# In some known directory oustide of any conflicting directories
mkdir  src/
cd src/
git clone https://github.com/opencv/opencv
git -C opencv checkout 4.x

# optionally
git clone https://github.com/opencv/opencv_contrib
git -C opencv_contrib checkout 4.x

# optionally
git clone https://github.com/opencv/opencv_extra
git -C opencv_extra checkout 4.x

mkdir build
cd build
cmake ../opencv

# Configure cmake:
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv
make
sudo make install
```


## Create a catkin workspace

Create a workspace in a meaningfull directory
```bash
mkdir -p ~/ws_bebop/src
cd ws_bebop/src
```

# Installation

Clone the package into your catkin workspace (in src folder): 
```bash
cd
cd ws_bebop
git clone  https://github.com/ src/ # TODO
# For off line development
# git fork https://github.com/ src/ # TODO
```

Compile
```bash
colcon build
# Don't forget to reload the environment
source ~/ws_bebop/install/setup.bash
```
Environment variables

```bash
export GZ_SIM_RESOURCE_PATH="$HOME/ws_bebop/src/ros2_bebop_ibvs/worlds:$HOME/ws_bebop/src/ros/models"
export GZ_VERSION=jetty
```



# Example

```bash
# screen 1
ros2 launch ros2_bebop_ibvs bebop1.launch.py
# screen 2
ros2 topic pub  /state std_msgs/Int32 "{data: 2}" --once #  TAKEOFF
ros2 topic pub  /state std_msgs/Int32 "{data: 3}" --once #  LAND
ros2 topic pub  /state std_msgs/Int32 "{data: 1}" --once #  IBVS
ros2 topic pub  /state std_msgs/Int32 "{data: 4}" --once #  STOP
```

<!-- END my ibvs  -->


### bebop_demo
Contains utility packages and demonstration setups:
- `set_pose`: Sets initial drone positions
- `setpoint`: Generates circular trajectories for single drone operation
- Various executable examples

### bebop_gui
Contains graphical user interfaces for the different demos.

### bebop_gz
Gazebo-specific components including:
- Drone models and plugins
- Simulation worlds
- A configurable world generator that adapts to N agents

### bebop_ros_gz
Integrates all Gazebo packages into a single project. This meta-package provides the complete interface for using ROS 2 with Gazebo simulation.
