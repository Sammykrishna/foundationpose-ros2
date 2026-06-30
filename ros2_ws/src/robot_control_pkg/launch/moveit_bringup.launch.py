import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # Use the official UR MoveIt2 launch — it handles:
    # - Correct URDF with ros2_control tags
    # - kinematics.yaml with KDL IK solver
    # - SRDF with planning groups and home position
    # - controller config
    # - move_group with OMPL planning pipeline
    ur_moveit_launch = IncludeLaunchDescription(
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

    return LaunchDescription([ur_moveit_launch])