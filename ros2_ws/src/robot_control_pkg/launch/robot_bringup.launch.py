import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_dir = get_package_share_directory('robot_control_pkg')
    ur_moveit_config = get_package_share_directory('ur_moveit_config')
    params_file = os.path.join(
        get_package_share_directory('pose_estimation_pkg'),
        'config', 'params.yaml'
    )

    # loads the URDF into robot_state_publisher, which broadcasts TF for every link
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_description'),
                'launch',
                'ur_description.launch.py'
            ])
        ]),
        launch_arguments={
            'ur_type': 'ur5e',
            'use_fake_hardware': 'true',
        }.items()
    )

    # move_group handles all planning requests using the robot's URDF and SRDF
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_moveit_config'),
                'launch',
                'ur_moveit.launch.py'
            ])
        ]),
        launch_arguments={
            'ur_type': 'ur5e',
            'use_fake_hardware': 'true',
            'launch_rviz': 'false',  # we use our own RViz2
        }.items()
    )

    # starts after MoveIt2 is ready
    grasp_executor = TimerAction(
        period=10.0,  # give MoveIt2 time to fully initialize
        actions=[
            Node(
                package='robot_control_pkg',
                executable='grasp_executor',
                name='grasp_executor',
                parameters=[params_file],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        robot_description_launch,
        moveit_launch,
        grasp_executor,
    ])