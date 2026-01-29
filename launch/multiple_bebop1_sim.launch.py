from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from string import Template
# from ros_gz_bridge.actions import RosGzBridge
import os
import yaml
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package routs
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_bebop_ibvs = get_package_share_directory('ros2_bebop_ibvs')

    #   Load bebop model
    bebop_model = os.path.join(pkg_bebop_ibvs,"models","parrot_bebop_2","model.sdf")
    with open(bebop_model, "r") as infp:
        robot_desc = infp.read()
    rd_template = Template(robot_desc) # convert string in template

    #   Configurations
    yaml_control = os.path.join(pkg_bebop_ibvs, 'config', 'ibfc_sim.yaml')
    with open(yaml_control, 'r') as file:
        config = yaml.safe_load(file)
    config["reference_image_prefix"] = os.path.join(pkg_bebop_ibvs, 'config', 'reference_f')
    if not os.path.exists(config["output"]):
        print(f"Output directory  {config["output"]} does not exists")
        return
    bridge_config = os.path.join(pkg_bebop_ibvs, 'config', 'bebopN.yaml')


    # Launch Gazebo
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': '-r -z 1000000 simple_lab_4_2.world ',
            # 'gz_args': '-r -z 1000000 simple_lab_2.world ',
        'on_exit_shutdown': 'true'}.items(),
    )


    bridge = []
    models = []
    controller = []
    # for i in range(1):
    for i in range(int(config["n_agents"])):
        #   Launch General Bridges
        ros_gz_bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name = 'parrot_bebop_2_'+str(i),
            output='screen',
            parameters=[{
                'expand_gz_topic_names': True,  # Activate the expand_gz_topic_names parameter
                'config_file': bridge_config,
            }],
        )
        bridge.append(ros_gz_bridge)

        # Launch image bridge
        ros_img_bridge = Node(
             package='ros_gz_image',
            executable='image_bridge',
            arguments=[f"/parrot_bebop_2_{i}/image"],
            output='screen',
        )
        bridge.append(ros_img_bridge)

        #   Spawn bebop
        rd_template = Template(robot_desc)
        bebop_spawn = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', 'parrot_bebop_2_'+str(i),
                '-world', 'default',
                '-string', rd_template.substitute(prefix=f"parrot_bebop_2_{i}"),
                '-x', str(-i),
                '-y', '-1.',
                '-z', '0.1',
                '-Y', '1.',
            ],
            output='screen',
        )
        models.append(bebop_spawn)


        # #   IBFC
        config["label"] = i
        _controller = Node(
                package='ros2_bebop_ibvs',
                executable='ibfc_sim',
                name='ibfc_'+str(i),
                output='screen',
                parameters=[config]
            )
        controller.append(_controller)

    return LaunchDescription([gz_sim] + models + bridge + controller)
