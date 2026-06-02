import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():

    pkg_sim = get_package_share_directory('simulation_pkg')
    pkg_pose = get_package_share_directory('pose_estimation_pkg')

    world_file = os.path.join(pkg_sim, 'worlds', 'table_scene.sdf')
    params_file = os.path.join(pkg_pose, 'config', 'params.yaml')
    rviz_config = os.path.join(pkg_sim, 'rviz', 'pose_estimation.rviz')

    # Path to your venv's site-packages
    # This tells ROS2 nodes where to find trimesh, torch, cv2 etc.
    venv_site_packages = (
        '/home/samanth-krishna/projects/ros2_ws/src/'
        'foundationpose-ros2/fpenv/lib/python3.12/site-packages'
    )

    # Get the current PYTHONPATH and prepend our venv packages to it
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_pythonpath = venv_site_packages + ':' + current_pythonpath

    # 1. Gazebo simulation
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v', '4', world_file],
        output='screen'
    )

    # 2. ROS-Gazebo bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_ros_bridge',
        arguments=[
            '/camera/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/camera/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen'
    )

    # 3. FoundationPose node — with venv PYTHONPATH injected
    pose_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='pose_estimation_pkg',
                executable='foundationpose_node',
                name='foundationpose_node',
                parameters=[params_file],
                output='screen',
                # This is the key fix — inject venv site-packages into
                # the node's Python path so it can find trimesh, torch etc.
                additional_env={'PYTHONPATH': new_pythonpath}
            )
        ]
    )

    # 4. RViz2
    rviz = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config],
                output='screen'
            )
        ]
    )

    return LaunchDescription([gazebo, bridge, pose_node, rviz])