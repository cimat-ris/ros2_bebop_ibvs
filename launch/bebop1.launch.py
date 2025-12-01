from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ros_gz_bridge.actions import RosGzBridge
from launch_ros.actions import Node
import os
import yaml
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Obtener rutas de los paquetes
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_bebop_ibvs = get_package_share_directory('ros2_bebop_ibvs')

    # Definir nombres de robots y condiciones iniciales como cadenas JSON
    robot_name = "bebop1"

    # Lanzar Gazebo
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': '-r -z 1000000 simple_lab_2.world ',
        'on_exit_shutdown': 'true'}.items(),
    )



    # Lanzar el puente ROS-Gazebo
    ros_gz_bridge = RosGzBridge(
        bridge_name='ros_gz_bridge',
        config_file=os.path.join(pkg_bebop_ibvs, 'config', 'bebop1.yaml'),
    )



    #   Spawn bebop
    bebop_model = os.path.join(pkg_bebop_ibvs,"models","parrot_bebop_2","model.sdf")
    bebop_spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'parrot_bebop_2',
            '-world', 'default',
            '-file', bebop_model,
            '-x', '-1.',
            '-y', '-1.',
            '-z', '0.1',
            '-Y', '1.',
        ],
        output='screen',
    )


    #   IBVS
    yaml_control = os.path.join(pkg_bebop_ibvs, 'config', 'control.yaml')
    with open(yaml_control, 'r') as file:
        config = yaml.safe_load(file)
    # config["ref_image"] = os.path.join(pkg_bebop_ibvs, 'config', 'reference_4_2.png')
    config["ref_image"] = os.path.join(pkg_bebop_ibvs, 'config', 'reference_2_2.png')
    # config["ref_image"] = os.path.join(pkg_bebop_ibvs, 'config', 'reference_1_aruco.png')
    if not os.path.exists(config["ref_image"]):
        print(f"Camara reference {config["ref_image"]} does not exists")
        return

    if not os.path.exists(config["output"]):
        print(f"Output directory  {config["output"]} does not exists")
        return

    controller = Node(
            package='ros2_bebop_ibvs',
            executable='ibvs',
            name='ibvs',
            output='screen',
            parameters=[config]
        )

    return LaunchDescription([
        gz_sim,
        bebop_spawn,
        ros_gz_bridge,
        controller,
    ])
