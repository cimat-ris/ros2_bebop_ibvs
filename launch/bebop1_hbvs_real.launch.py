from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ros_gz_bridge.actions import RosGzBridge
from launch_ros.actions import Node
from string import Template
import os
import yaml
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    #   package route
    pkg_bebop_ibvs = get_package_share_directory('ros2_bebop_ibvs')

    #   Configs
    yaml_control = os.path.join(pkg_bebop_ibvs, 'config', 'hbvs_real.yaml')
    with open(yaml_control, 'r') as file:
        config = yaml.safe_load(file)
    # config["ref_image"] = os.path.join(pkg_bebop_ibvs, 'config', 'real_1.png')
    # config["ref_image"] = os.path.join(pkg_bebop_ibvs, 'config', 'real_2.png')
    config["ref_image"] = os.path.join(pkg_bebop_ibvs, 'config', 'real_3.png')
    if not os.path.exists(config["ref_image"]):
        print(f"Camara reference {config["ref_image"]} does not exists")
        return

    if not os.path.exists(config["output"]):
        print(f"Output directory  {config["output"]} does not exists")
        return

    #   HBVS
    controller = Node(
            package='ros2_bebop_ibvs',
            executable='hbvs_real',
            name='hbvs',
            output='screen',
            parameters=[config]
        )

    return LaunchDescription([
        controller,
    ])
