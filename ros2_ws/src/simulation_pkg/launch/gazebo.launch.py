import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution



def generate_launch_description():

    pkg_sim = get_package_share_directory('simulation_pkg')
    pkg_pose = get_package_share_directory('pose_estimation_pkg')

    world_file = os.path.join(pkg_sim, 'worlds', 'table_scene.sdf')
    params_file = os.path.join(pkg_pose, 'config', 'params.yaml')
    rviz_config = os.path.join(
        get_package_share_directory('simulation_pkg'),
        'rviz', 'pose_estimation.rviz'
    )

    venv_site_packages = (
        '/home/samanth-krishna/projects/ros2_ws/src/'
        'foundationpose-ros2/fpenv/lib/python3.12/site-packages'
    )

    sam2_path = '/home/samanth-krishna/projects/ros2_ws/src/foundationpose-ros2/sam2'

    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_pythonpath = venv_site_packages + ':' + sam2_path + ':' + current_pythonpath

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-s', '-r', '-v', '4', world_file],
        output='screen',
        additional_env={
            '__NV_PRIME_RENDER_OFFLOAD': '1',
            '__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
            '__EGL_VENDOR_LIBRARY_FILENAMES':
                '/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
            'DRI_PRIME': '1',
            'DISPLAY': os.environ.get('DISPLAY', ':1'),
        }
    )

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

    # SAM2 starts at 4s so its mask is ready before FoundationPose inits
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

    rviz = TimerAction(
        period=5.0,
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

    scene_markers = Node(
        package='simulation_pkg',
        executable='scene_markers_node',
        name='scene_markers_node',
        output='screen'
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_base_link',
        arguments=['--x', '-0.1', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'world',
                   '--child-frame-id', 'base_link'],
        output='screen'
    )

    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_camera',
        arguments=['--x', '0.8', '--y', '-0.5', '--z', '1.45',
                   '--roll', '0', '--pitch', '0.9195', '--yaw', '1.5708',
                   '--frame-id', 'world',
                   '--child-frame-id', 'realsense_d435i/link/color_camera'],
        output='screen'
    )

    static_tf_camera_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_to_optical',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '0.5', '--qy', '-0.5', '--qz', '0.5', '--qw', '-0.5',
                   '--frame-id', 'realsense_d435i/link/color_camera',
                   '--child-frame-id', 'realsense_d435i/link/color_camera_optical'],
        output='screen'
    )

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

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content, 'use_sim_time': False}]
    )

    # zeros put the UR5e in an upright home configuration
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{
            'rate': 10,
            'zeros': {
                'shoulder_pan_joint': 0.0,
                'shoulder_lift_joint': -1.5708,
                'elbow_joint': 1.5708,
                'wrist_1_joint': -1.5708,
                'wrist_2_joint': 0.0,
                'wrist_3_joint': 0.0,
            }
        }]
    )

    return LaunchDescription([
        gazebo,
        bridge,
        static_tf,
        static_tf_camera,
        static_tf_camera_optical,
        robot_state_publisher,
        joint_state_publisher,
        scene_markers,
        sam2_node,
        pose_node,
        rviz,
    ])