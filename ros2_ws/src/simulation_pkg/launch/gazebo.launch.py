import os
from ament_index_python.packages import get_package_share_directory # type: ignore
from launch import LaunchDescription # type: ignore
from launch.actions import ExecuteProcess, TimerAction # type: ignore
from launch_ros.actions import Node # type: ignore


def generate_launch_description():

    # Find where our package is installed so we can locate the world file
    pkg_dir = get_package_share_directory('simulation_pkg')
    world_file = os.path.join(pkg_dir, 'worlds', 'table_scene.sdf')

    # 1. Start Gazebo Harmonic with our world file
    # -r means "start running immediately" (don't pause on startup)
    # -v 4 means verbose logging level 4 (useful for debugging sensor issues)
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v', '4', world_file],
        output='screen'
    )

    # 2. ros_gz_bridge: this is the translator between Gazebo and ROS2
    # Gazebo uses its own internal message system; ROS2 uses its own.
    # The bridge subscribes to Gazebo topics and republishes them as ROS2 topics.
    # Format: gazebo_topic@ros2_msg_type[gz_msg_type
    # The [ means "Gazebo -> ROS2 direction"
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_ros_bridge',
        arguments=[
            # RGB image: Gazebo Image -> ROS2 sensor_msgs/Image
            '/camera/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            # Depth image: Gazebo Image -> ROS2 sensor_msgs/Image
            '/camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            # Camera info (intrinsics like focal length, principal point)
            # FoundationPose needs this to back-project depth into 3D points
            '/camera/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/camera/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            # Clock topic so ROS2 nodes use simulation time, not wall clock time
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen'
    )

    # 3. RViz2 for visualization
    # We delay it by 3 seconds to give Gazebo time to start first
    rviz = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen'
            )
        ]
    )

    return LaunchDescription([gazebo, bridge, rviz])