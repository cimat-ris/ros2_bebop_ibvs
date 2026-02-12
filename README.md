# Requirements

## Installing ROS Rolling on ubuntu 24.04 LTS (Noble Numbat)

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

sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
 curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
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

Install opencv from source (version 4.x):
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

mkdir build
cd build

cmake ../opencv
# Configure cmake:
# Any other needed changes can be done with the following command
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv

#   build: (can be made in parallel using -j )
make
# make -j$(nproc)

#   Install
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

# Installation

## Create a catkin workspace (if using a new one)

Create a workspace in a meaningfull directory
```bash
mkdir -p ~/ws_bebop/src
cd ws_bebop/src
```

Clone the package into your catkin workspace (in src folder): 
```bash
cd
cd ws_bebop
git clone https://github.com/cimat-ris/ros2_bebop_ibvs.git src/
# For off line development
# git fork https://github.com/cimat-ris/ros2_bebop_ibvs.git src/
```

Compile
```bash
colcon build
# Don't forget to reload the environment
source ~/ws_bebop/install/setup.bash
```

Environment variables (have to be defined for each session)
```bash
export GZ_SIM_RESOURCE_PATH="$HOME/ws_bebop/src/ros2_bebop_ibvs/worlds:$HOME/ws_bebop/src/ros/models:"
export GZ_VERSION=jetty
```



# Simulation or a single agent IBVS

```bash
# screen 1
ros2 launch ros2_bebop_ibvs bebop1_sim.launch.py
# screen 2
ros2 run rqt_image_view rqt_image_view
# screen 3
ros2 topic pub  /state std_msgs/Int32 "{data: 2}" --once #  TAKEOFF
ros2 topic pub  /state std_msgs/Int32 "{data: 3}" --once #  LAND
ros2 topic pub  /state std_msgs/Int32 "{data: 1}" --once #  IBVS
ros2 topic pub  /state std_msgs/Int32 "{data: 4}" --once #  STOP
ros2 topic pub  /state std_msgs/Int32 "{data: 5}" --once #  INITAL CONDITION
```

For simplicity the following aliases can be defined
```bash
alias takeoff="ros2 topic pub  /state std_msgs/Int32 \"{data: 2}\" --once" #  TAKEOFF
alias land="ros2 topic pub  /state std_msgs/Int32 \"{data: 3}\" --once" #  LAND
alias ibvs="ros2 topic pub  /state std_msgs/Int32 \"{data: 1}\" --once" #  IBVS
alias stop="ros2 topic pub  /state std_msgs/Int32 \"{data: 4}\" --once" #  STOP
alias init="ros2 topic pub  /state std_msgs/Int32 \"{data: 5}\" --once" #  INITAL CONDITION
```


# Simulation of multiple agents IBVS - Formation Control

```bash
# screen 1
ros2 launch   ros2_bebop_ibvs multiple_bebop1_sim.launch.py
# screen 2
ros2 run rqt_image_view rqt_image_view
# screen 3
ros2 topic pub --once  /state std_msgs/Int32 "{data: 2}"  #  TAKEOFF
ros2 topic pub --once  /state std_msgs/Int32 "{data: 3}"  #  LAND
ros2 topic pub --once  /state std_msgs/Int32 "{data: 1}"  #  IBVS
ros2 topic pub --once  /state std_msgs/Int32 "{data: 4}"  #  STOP
ros2 topic pub --once  /state std_msgs/Int32 "{data: 5}"  #  INITAL CONDITION
```

For simplicity the following aliases can be defined
```bash
alias takeoff="ros2 topic pub --once /state std_msgs/Int32 \"{data: 2}\" " #  TAKEOFF
alias land="ros2 topic pub --once /state std_msgs/Int32 \"{data: 3}\" " #  LAND
alias ibvs="ros2 topic pub --once /state std_msgs/Int32 \"{data: 1}\" " #  IBFC
alias stop="ros2 topic pub --once /state std_msgs/Int32 \"{data: 4}\" " #  STOP
alias init="ros2 topic pub --once /state std_msgs/Int32 \"{data: 5}\" " #  INITAL CONDITION
```


#   IBVS with real Bebop


```bash
#   Screen 1
ros2 launch ros2_bebop_driver bebop_node_launch.xml ip:="192.168.45.1"
ros2 launch ros2_bebop_driver bebop_node_launch.xml ip:="192.168.42.1"

#   Screen 2 (Emergency stop)
ros2 topic pub --once bebop/reset std_msgs/Empty
ros2 topic pub --once bebop/land std_msgs/Empty

#   Screen 3
ros2 run rqt_image_view rqt_image_view

#   Screen 4
ros2 launch ros2_bebop_ibvs bebop1_real.launch.py

#   Screen 5
ros2 topic pub --once /stop std_msgs/Empty
ros2 topic pub --once /start std_msgs/Empty # After starting the launch
```

Disable camera stabilization
```bash


# Reset driver:
##  Connect (fast press powerbutton 4 times )
telnet 192.168.45.1
telnet 192.168.42.1
##  Search Driver ID
ps | grep dragon
##  Save Driver ID
ID_DRAGON=$(ps | grep dragon | head -n1 | awk '{print $1}')
#  Detener dirver
kill -9 ${ID_DRAGON}
sleep 1
##  Confirmar
ps | grep dragon
# Reiniciar driver
/usr/bin/dragon-prog -S 0 &
```


#   Plot data

The following utility plots:
+ State evolution
+ 3D Plot of the state evolution
+ Control velocities
+ Image feature error
+ Image feature evolution in a planar projection
+ SVD singular value evolution if saved

Arguments:
+ directory (non optional) = were data is stored and output will be placed
+ reference = reference image
+ pose = 4 coordinate state for the pose where the reference is taken
+ markers = fiducial marker family to be used in the corner extraction

To run:
```
$ python3 scripts/plot_data.py -h
usage: python3 miguel_iros.py [-h] [--reference REFERENCE] [--pose POSE POSE POSE POSE]
                              [--markers {4X4_50,4X4_100,4X4_250,4X4_1000,5X5_50,5X5_100,5X5_250,5X5_1000,6X6_50,6X6_100,6X6_250,6X6_1000,7X7_50,7X7_100,7X7_250,7X7_1000,ARUCO_ORIGINAL,APRILTAG_16h5,APRILTAG_25h9,APRILTAG_36h10,APRILTAG_36h11,ARUCO_MIP_36h12}]
                              directory

Plotting experiment data

positional arguments:
  directory             Directory name (default output/)

options:
  -h, --help            show this help message and exit
  --reference REFERENCE
                        IBVS reference image (default config/reference.png)
  --pose POSE POSE POSE POSE
                        Reference pose [x, y, z, yaw (degs)] default [1., 0., 1., 90]
  --markers {4X4_50,4X4_100,4X4_250,4X4_1000,5X5_50,5X5_100,5X5_250,5X5_1000,6X6_50,6X6_100,6X6_250,6X6_1000,7X7_50,7X7_100,7X7_250,7X7_1000,ARUCO_ORIGINAL,APRILTAG_16h5,APRILTAG_25h9,APRILTAG_36h10,APRILTAG_36h11,ARUCO_MIP_36h12}
                        Fiducial markers family (default 6X6_1000)

```
