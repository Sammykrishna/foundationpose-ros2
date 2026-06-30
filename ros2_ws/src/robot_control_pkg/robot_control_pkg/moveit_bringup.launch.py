import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch.substitutions import (
    Command, FindExecutable, PathJoinSubstitution
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # Build robot description from xacro
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        '/opt/ros/jazzy/share/ur_description/urdf/ur.urdf.xacro',
        ' ',
        'ur_type:=ur5e',
        ' ',
        'name:=ur',
        ' ',
        'use_fake_hardware:=true',
        ' ',
        'fake_sensor_commands:=false',
        ' ',
        'sim_gazebo:=false',
    ])

    robot_description = {'robot_description': robot_description_content}

    # MoveIt2 configuration from ur_moveit_config
    robot_description_semantic_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([
            FindPackageShare('ur_moveit_config'),
            'srdf', 'ur.srdf.xacro'
        ]),
        ' ',
        'name:=ur',
        ' ',
        'ur_type:=ur5e',
    ])

    robot_description_semantic = {
        'robot_description_semantic': robot_description_semantic_content
    }

    # MoveIt2 move_group node — the core planning server
    # This is what accepts planning requests and returns trajectories
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            {'use_sim_time': False},
            {'publish_robot_description_semantic': True},
        ]
    )

    # ros2_control node — manages the fake hardware controllers
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            robot_description,
            PathJoinSubstitution([
                FindPackageShare('ur_robot_driver'),
                'config', 'ur_controllers.yaml'
            ])
        ],
        output='screen'
    )

    # Spawn joint state broadcaster
    joint_state_broadcaster_spawner = TimerAction(
        period=2.0,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen'
        )]
    )

    # Spawn joint trajectory controller
    joint_trajectory_controller_spawner = TimerAction(
        period=3.0,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['scaled_joint_trajectory_controller'],
            output='screen'
        )]
    )

    return LaunchDescription([
        move_group_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        joint_trajectory_controller_spawner,
    ])