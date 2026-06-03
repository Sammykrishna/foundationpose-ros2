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

    # 3. SAM2 node — starts at 4s so its mask is ready before FoundationPose inits
    sam2_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='pose_estimation_pkg',
                executable='sam2_node',
                name='sam2_node',
                parameters=[params_file],
                output='screen',
                additional_env={'PYTHONPATH': new_pythonpath}
            )
        ]
    )

    # 4. FoundationPose node — delayed to 7s to give SAM2 a 3-second head start
    pose_node = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='pose_estimation_pkg',
                executable='foundationpose_node',
                name='foundationpose_node',
                parameters=[params_file],
                output='screen',
                additional_env={'PYTHONPATH': new_pythonpath}
            )
        ]
    )

    # 5. RViz2
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

    return LaunchDescription([gazebo, bridge, sam2_node, pose_node, rviz])