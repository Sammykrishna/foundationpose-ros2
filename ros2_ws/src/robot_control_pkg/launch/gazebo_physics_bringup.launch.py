"""Physics-based manipulation bringup (Milestone: option 1).

Unlike moveit_bringup/demo.launch.py (mock_components/GenericSystem, instant
fake actuation, no contact physics), this spawns the real robot model into
Gazebo and actuates it through the gz_ros2_control plugin, so the gripper
has to actually contact and hold the box for a grasp to count as
successful, instead of MoveIt's planning scene deciding "attached" by
bookkeeping alone.

Usage:
    ros2 launch robot_control_pkg gazebo_physics_bringup.launch.py
    ros2 run robot_control_pkg planning_scene_manager
    ros2 run robot_control_pkg grasp_executor
"""
import os
import re
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


# The pads are rigid on the finger, so they tilt with the knuckle when it
# closes (the real gripper keeps them parallel with a four-bar linkage).
# Pre-rotating each pad by the knuckle angle it reaches on this box makes it
# vertical at the grasp, so it grips on its face rather than an edge.
PAD_CLOSED_TILT = 0.19


def _robot_state_publisher_action(context, *args, **kwargs):
    """Build robot_description with a real subprocess call (not a launch
    Command substitution) so the raw URDF text can be patched before
    publishing: ur_description hardcodes damping="0" friction="0" on every
    arm joint with no xacro arg to override it, and gz_ros2_control's
    position_proportional_gain servo has no separate derivative/damping
    term of its own -- together that's an undamped spring against gravity,
    which oscillated indefinitely instead of settling in testing. Patching
    in real joint damping here is what makes the arm actually hold still.
    ros_gz_sim spawns from this same published /robot_description topic,
    so the fix reaches Gazebo automatically without touching the spawn step.
    """
    pkg_moveit_config = get_package_share_directory('ur5e_robotiq_moveit_config')
    xacro_path = os.path.join(pkg_moveit_config, 'config', 'ur.urdf.xacro')
    initial_positions_file = os.path.join(
        pkg_moveit_config, 'config', 'initial_positions_gazebo.yaml')

    raw_urdf = subprocess.check_output([
        'xacro', xacro_path,
        f'initial_positions_file:={initial_positions_file}',
        'sim_gazebo:=true',
    ]).decode()
    patched_urdf = raw_urdf.replace(
        '<dynamics damping="0" friction="0"/>',
        '<dynamics damping="5.0" friction="1.0"/>',
    )

    # Gazebo can't resolve package:// URIs (no GZ_SIM_RESOURCE_PATH), so
    # the UR arm's visual meshes silently didn't render; make them absolute.
    def _abs_uri(m):
        return f'file://{get_package_share_directory(m.group(1))}/'
    patched_urdf = re.sub(r'package://([A-Za-z0-9_]+)/', _abs_uri, patched_urdf)

    # dartsim can't build ANY mesh collision shape at all (see the physics
    # engine note above), so the two fingertip links -- the only ones that
    # actually need to touch the box for a real grasp -- get replaced with
    # a primitive box matching the collision STL's true bounding box
    # (measured directly from the binary STL: size 0.0312x0.0270x0.0570,
    # centered at the offsets below relative to each fingertip link
    # origin). The arm's own mesh collisions stay dropped under dartsim;
    # that's fine since MoveIt's own mesh-capable collision checking is
    # what actually keeps the arm out of the table, not Gazebo's.
    # The fingertip links hang off passive mimic joints with no damping, and
    # in Gazebo they were measured swinging to wrong, asymmetric poses (right
    # tip ~9cm off), dragging the pad collision boxes into the box. So the
    # pad collision goes on the rigid finger_link instead (fixed to its
    # knuckle, which does track the mimic correctly), offset to where the
    # tip pad sits at the open pose, and the passive joints get damping.
    for side, sx, cx in (('left', 1.0, -0.00965), ('right', -1.0, 0.00965)):
        x = sx * 0.00563134 + cx
        z = 0.04718515 + 0.0225
        collision = (
            '<collision>'
            f'<origin rpy="0 {sx * PAD_CLOSED_TILT} 0" xyz="{x} 0.0 {z}"/>'
            '<geometry><box size="0.0312 0.0270 0.0570"/></geometry>'
            '</collision>'
        )
        pattern = re.compile(
            r'(<link name="robotiq_85_' + side + r'_finger_link">.*?)(</link>)',
            re.S)
        patched_urdf, n = pattern.subn(
            lambda m: m.group(1) + collision + m.group(2), patched_urdf, count=1)
        if n != 1:
            raise RuntimeError(f"{side} finger_link not found for pad collision")

    # The passive tip joints don't hold their mimic angle in dartsim (right
    # tip measured ~4cm out of place at the open pose); freeze them rigid.
    for side in ('left', 'right'):
        pattern = re.compile(
            r'<joint name="robotiq_85_' + side + r'_finger_tip_joint" type="continuous">(.*?)</joint>',
            re.S)
        def _fix(m):
            body = re.sub(r'<(axis|mimic|dynamics)[^>]*/>', '', m.group(1))
            return (f'<joint name="robotiq_85_{side}_finger_tip_joint" type="fixed">'
                    + body + '</joint>')
        patched_urdf, n = pattern.subn(_fix, patched_urdf, count=1)
        if n != 1:
            raise RuntimeError(f"{side} finger_tip_joint not found")

    patched_urdf = re.sub(
        r'(<joint name="robotiq_85_right_knuckle_joint".*?)<mimic[^>]*/>',
        r'\1', patched_urdf, count=1, flags=re.S)

    for jn in ('left_inner_knuckle', 'right_inner_knuckle'):
        pattern = re.compile(
            r'(<joint name="robotiq_85_' + jn + r'_joint"[^>]*>.*?)(</joint>)', re.S)
        patched_urdf, n = pattern.subn(
            lambda m: m.group(1) + '<dynamics damping="0.05" friction="0.01"/>' + m.group(2),
            patched_urdf, count=1)
        if n != 1:
            raise RuntimeError(f"joint {jn} not found for damping")

    # Contact sensors on the box showed that the robot's mesh collisions ARE
    # active in Gazebo and misbehave: the passive inner-knuckle meshes droop
    # onto the box top, the gripper-base and fingertip meshes touch the box
    # from the pre-grasp pose on, and together they wedge on the box's top
    # edges and stop the descent ~8 cm short. Keep only the two fingertip pad
    # boxes as Gazebo collisions; MoveIt's own model keeps the arm off the table.
    def _keep_pads_only(m):
        return m.group(0) if 'size="0.0312 0.0270 0.0570"' in m.group(0) else ''
    patched_urdf = re.sub(r'<collision[^>]*>.*?</collision>', _keep_pads_only,
                          patched_urdf, flags=re.S)

    friction = ''.join(
        f'<gazebo reference="robotiq_85_{side}_knuckle_link">'
        '<mu1>3.0</mu1><mu2>3.0</mu2></gazebo>' for side in ('left', 'right'))
    patched_urdf = patched_urdf.replace('</robot>', friction + '</robot>')

    robot_description = {
        'robot_description': ParameterValue(patched_urdf, value_type=str)
    }
    return [Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )]


def generate_launch_description():
    pkg_sim = get_package_share_directory('simulation_pkg')
    world_file = os.path.join(pkg_sim, 'worlds', 'table_scene.sdf')

    use_rviz_arg = DeclareLaunchArgument('use_rviz', default_value='false')
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='true = gz sim -s (server only). false (default) = shows the '
                    'Gazebo GUI, so the grasp/lift is actually visible.'
    )

    robot_state_publisher = OpaqueFunction(function=_robot_state_publisher_action)

    # dartsim (gz sim's default physics engine) can't build collision shapes
    # from mesh geometry at all ("Mesh construction ... not implemented for
    # dartsim"), which silently drops every mesh-based collision on this
    # robot (arm links and gripper fingers alike). bullet-featherstone does
    # support mesh collision, but testing found its joint velocity/motor
    # support is unreliable here: joint_trajectory_controller reports
    # "Goal successfully reached!" while the real simulated joint barely
    # moves, matching a known feature-parity gap
    # (gazebosim/gz-physics#545, #1087, gazebosim/gz-sim#2729). Staying on
    # dartsim for reliable actuation and instead giving just the two
    # fingertip links (the only ones that need to physically contact the
    # box) primitive box collision in the patched URDF below is the more
    # robust trade: MoveIt's own (mesh-capable) collision checking already
    # keeps the arm itself away from the table without Gazebo needing to
    # re-check that at the physics level.
    # Same NVIDIA offload env gazebo.launch.py uses: without it the depth
    # sensor renders on the Mesa fallback and returns mostly invalid depth.
    GZ_ENV = {
        '__NV_PRIME_RENDER_OFFLOAD': '1',
        '__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
        '__EGL_VENDOR_LIBRARY_FILENAMES': '/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
        'DRI_PRIME': '1',
        'DISPLAY': os.environ.get('DISPLAY', ':1'),
    }
    gz_args = ['-r', '-v', '4', world_file]
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim'] + ['-s'] + gz_args,
        output='screen',
        condition=IfCondition(LaunchConfiguration('headless')),
        additional_env=GZ_ENV,
    )
    gazebo_gui = ExecuteProcess(
        cmd=['gz', 'sim'] + gz_args,
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('headless'), "' == 'false'"])
        ),
        additional_env=GZ_ENV,
    )

    # gz_ros2_control's controller_manager lives inside the Gazebo process
    # itself once the model is spawned; there's no standalone ros2_control_node.
    #
    # Timing here matters a lot more than it looks: with a several-second
    # gap between spawn and controller activation, the arm free-falls
    # uncommanded that whole time (hold_joints wasn't enough to stop it in
    # testing), and joint_trajectory_controller just captures whatever
    # mid-fall position it happens to be at when it activates and holds
    # THAT forever, regardless of position_proportional_gain, initial_value,
    # or any other tuning -- none of that ever gets a chance to act before
    # the fall already produced a table-colliding pose. Keeping this gap
    # under a second is what actually fixed it, not any of the gain values.
    spawn_robot = TimerAction(
        period=2.0,
        actions=[Node(
            package='ros_gz_sim',
            executable='create',
            arguments=['-topic', 'robot_description', '-name', 'ur5e_robotiq',
                       '-x', '0', '-y', '0', '-z', '0'],
            output='screen',
        )]
    )

    joint_state_broadcaster_spawner = TimerAction(
        period=2.5,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen',
        )]
    )
    ur_manipulator_controller_spawner = TimerAction(
        period=2.7,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['ur_manipulator_controller'],
            output='screen',
        )]
    )
    gripper_effort_forward_spawner = TimerAction(
        period=2.9,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['gripper_controller'],
            output='screen',
        )]
    )
    # The actual open/close/stall-detection logic; presents the same
    # gripper_controller/gripper_cmd action MoveIt already expects, so
    # grasp_executor doesn't need to know this isn't a real ros2_control
    # controller underneath.
    gripper_effort_controller = TimerAction(
        period=3.5,
        actions=[Node(
            package='robot_control_pkg',
            executable='gripper_effort_controller',
            output='screen',
        )]
    )

    pkg_moveit_config = get_package_share_directory('ur5e_robotiq_moveit_config')
    moveit_config = (
        MoveItConfigsBuilder("ur", package_name="ur5e_robotiq_moveit_config")
        .robot_description(mappings={
            "sim_gazebo": "true",
            "initial_positions_file": os.path.join(
                pkg_moveit_config, 'config', 'initial_positions_gazebo.yaml'),
        })
        .to_moveit_configs()
    )
    move_group_ld = generate_move_group_launch(moveit_config)

    rviz_config = os.path.join(pkg_sim, 'rviz', 'pose_estimation.rviz')
    rviz = TimerAction(
        period=10.0,
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        )]
    )

    # Bridges/TFs that gazebo.launch.py used to provide (the /clock bridge in
    # particular: every sim-time node freezes without it).
    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='gz_ros_bridge',
        arguments=[
            '/camera/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/camera/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen')
    static_tf_camera = Node(
        package='tf2_ros', executable='static_transform_publisher', name='world_to_camera',
        arguments=['--x', '0.8', '--y', '0.5', '--z', '1.45', '--roll', '0',
                   '--pitch', '0.8858', '--yaw', '-1.5708', '--frame-id', 'world',
                   '--child-frame-id', 'realsense_d435i/link/color_camera'])
    static_tf_optical = Node(
        package='tf2_ros', executable='static_transform_publisher', name='camera_to_optical',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--qx', '0.5', '--qy', '-0.5',
                   '--qz', '0.5', '--qw', '-0.5',
                   '--frame-id', 'realsense_d435i/link/color_camera',
                   '--child-frame-id', 'realsense_d435i/link/color_camera_optical'])
    scene_markers = Node(package='simulation_pkg', executable='scene_markers_node',
                         name='scene_markers_node', output='screen')

    pkg_pose = get_package_share_directory('pose_estimation_pkg')
    params_file = os.path.join(pkg_pose, 'config', 'params.yaml')
    fp_root = '/home/samanth-krishna/projects/ros2_ws/src/foundationpose-ros2'
    pyenv = {'PYTHONPATH': f"{fp_root}/fpenv/lib/python3.12/site-packages:{fp_root}/sam2:"
                           + os.environ.get('PYTHONPATH', '')}
    perception = LaunchConfiguration('perception')
    sam2_node = TimerAction(period=6.0, actions=[Node(
        package='pose_estimation_pkg', executable='sam2_node', name='sam2_node',
        parameters=[params_file], output='screen', additional_env=pyenv,
        condition=IfCondition(perception))])
    pose_node = TimerAction(period=9.0, actions=[Node(
        package='pose_estimation_pkg', executable='foundationpose_node',
        name='foundationpose_node', parameters=[params_file], output='screen',
        additional_env=pyenv, condition=IfCondition(perception))])

    # mission:=true also starts the planning scene manager and the grasp
    # executor, which then runs the whole pick, place, re-detect, return demo.
    mission = LaunchConfiguration('mission')
    scene_manager = TimerAction(period=30.0, actions=[Node(
        package='robot_control_pkg', executable='planning_scene_manager',
        output='screen', condition=IfCondition(mission))])
    executor = TimerAction(period=32.0, actions=[Node(
        package='robot_control_pkg', executable='grasp_executor', output='screen',
        parameters=[{'sim_gazebo': True, 'mission': True, 'single_shot': True}],
        condition=IfCondition(mission))])

    ld = LaunchDescription([
        DeclareLaunchArgument('mission', default_value='false'),
        scene_manager, executor,
        DeclareLaunchArgument('perception', default_value='false'),
        bridge, static_tf_camera, static_tf_optical, scene_markers,
        sam2_node, pose_node,
        use_rviz_arg,
        headless_arg,
        # gz_ros2_control's controller_manager (running inside the Gazebo
        # process) stamps /joint_states with Gazebo's simulation clock, not
        # wall time. Every node below that needs to reason about "how
        # fresh is this state" -- trajectory_execution_manager above all
        # -- must be on that same clock or every freshness check just
        # compares two unrelated numbers and always fails. For
        # FollowJointTrajectory (the arm) that failure is only a WARN and
        # execution proceeds anyway; for GripperCommand (the gripper) it's
        # fatal and the goal is never even sent -- MoveIt just reports the
        # execute() call as ABORTED, and grasp_executor never checked that
        # return value, so the whole thing looked like a silent no-op.
        # SetParameter applies to every Node in this LaunchDescription,
        # including move_group's entities appended below.
        SetParameter(name='use_sim_time', value=True),
        robot_state_publisher,
        gazebo,
        gazebo_gui,
        spawn_robot,
        joint_state_broadcaster_spawner,
        ur_manipulator_controller_spawner,
        gripper_effort_forward_spawner,
        gripper_effort_controller,
        rviz,
    ])
    for entity in move_group_ld.entities:
        ld.add_action(entity)
    return ld
