#!/usr/bin/env python3
"""ROS2 node that subscribes to /object_pose from FoundationPose, computes
a grasp pose above the object, and uses MoveIt2 to plan and execute a
collision-free pick and lift trajectory for the UR5e with a Robotiq 2F-85
gripper."""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
import numpy as np

from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from std_msgs.msg import String, Empty
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration as DurationMsg
from visualization_msgs.msg import Marker, MarkerArray
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from control_msgs.action import GripperCommand, FollowJointTrajectory

from robot_control_pkg.box_geometry_utils import (
    get_box_center_from_bottom_pose,
    SUGAR_BOX_HALF_HEIGHT,
)

# MoveIt2 Python bindings
try:
    from moveit.planning import MoveItPy
    from moveit_configs_utils import MoveItConfigsBuilder
    MOVEIT_AVAILABLE = True
except ImportError:
    print("[WARN] MoveIt2 Python bindings not available, using mock mode")
    MOVEIT_AVAILABLE = False

from scipy.spatial.transform import Rotation


class GraspState:
    IDLE = "IDLE"
    MOVING_HOME = "MOVING_HOME"
    OPENING_GRIPPER = "OPENING_GRIPPER"
    PRE_GRASP = "PRE_GRASP"
    GRASPING = "GRASPING"
    CLOSING_GRIPPER = "CLOSING_GRIPPER"
    LIFTING = "LIFTING"
    RETURNING_HOME = "RETURNING_HOME"
    TRANSPORTING = "TRANSPORTING"
    PLACING = "PLACING"
    RELEASING = "RELEASING"
    RETREATING = "RETREATING"
    VERIFYING_PLACE = "VERIFYING_PLACE"
    REDETECTING = "REDETECTING"
    DONE = "DONE"
    ERROR = "ERROR"


class GraspExecutorNode(Node):
    def __init__(self):
        super().__init__('grasp_executor')

        self.declare_parameter('planning_group', 'ur_manipulator')
        self.declare_parameter('gripper_group', 'gripper')
        # tcp_link accounts for the adapter plate and gripper reach, tool0 does not
        self.declare_parameter('end_effector_link', 'tcp_link')
        self.declare_parameter('pre_grasp_height', 0.15)
        self.declare_parameter('pose_stability_count', 10)
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('max_velocity_scaling', 0.3)
        # Must match whatever gazebo_physics_bringup.launch.py spawned the
        # robot with. grasp_executor builds its own separate MoveIt robot
        # model (below) independent of that launch file's; if this is left
        # false while the real hardware is running with sim_gazebo:=true,
        # MoveIt's controller dispatch believes the gripper is still
        # position-commanded and silently skips sending gripper goals at
        # all -- no error, "Opening/Closing gripper..." just logs and the
        # sequence moves on with the gripper never actually commanded.
        self.declare_parameter('sim_gazebo', False)
        # true = stop after the first grasp attempt instead of re-arming.
        self.declare_parameter('single_shot', False)
        # Carry the grasped box to (pick xy + place_dx/dy), set it down,
        # release, retreat home, then confirm with live perception.
        self.declare_parameter('place_enabled', False)
        self.declare_parameter('place_dx', 0.0)
        self.declare_parameter('place_dy', 0.20)
        self.declare_parameter('place_tolerance', 0.04)
        self.declare_parameter('verify_timeout_sec', 60.0)
        # How far above the grasp height the box is carried (the "full" lift).
        self.declare_parameter('lift_height', 0.30)
        # Two-leg demo: pick from home, put it down at a random spot, then
        # find it again with SAM2 + FoundationPose, pick it up and return it home.
        self.declare_parameter('mission', False)
        self.declare_parameter('slow_descent', False)
        self.declare_parameter('descent_duration_sec', 8.0)
        self.declare_parameter('home_x', 0.80)
        self.declare_parameter('home_y', 0.0)
        self.declare_parameter('random_seed', -1)
        self.declare_parameter('place_x_min', 0.65)
        self.declare_parameter('place_x_max', 0.95)
        self.declare_parameter('place_y_min', 0.10)
        self.declare_parameter('place_y_max', 0.25)
        self.declare_parameter('min_place_separation', 0.12)

        self.planning_group = self.get_parameter('planning_group').value
        self.gripper_group = self.get_parameter('gripper_group').value
        self.eef_link = self.get_parameter('end_effector_link').value
        self.pre_grasp_height = self.get_parameter('pre_grasp_height').value
        self.stability_count = self.get_parameter('pose_stability_count').value
        self.planning_time = self.get_parameter('planning_time').value
        self.velocity_scale = self.get_parameter('max_velocity_scaling').value
        self.sim_gazebo = self.get_parameter('sim_gazebo').value
        self.single_shot = self.get_parameter('single_shot').value
        self.place_enabled = self.get_parameter('place_enabled').value or self.get_parameter('mission').value
        self.place_dx = self.get_parameter('place_dx').value
        self.place_dy = self.get_parameter('place_dy').value
        self.place_tolerance = self.get_parameter('place_tolerance').value
        self.verify_timeout = self.get_parameter('verify_timeout_sec').value
        self.place_result = None
        self.lift_height = self.get_parameter('lift_height').value
        self.mission = self.get_parameter('mission').value
        self.slow_descent = self.get_parameter('slow_descent').value
        self.descent_duration = self.get_parameter('descent_duration_sec').value
        self.home_x = self.get_parameter('home_x').value
        self.home_y = self.get_parameter('home_y').value
        seed = self.get_parameter('random_seed').value
        self.rng = np.random.default_rng(None if seed < 0 else seed)
        self.collect_poses = None

        self.get_logger().info("Grasp executor starting...")

        self.current_state = GraspState.IDLE
        self.latest_pose = None
        self.pose_buffer = []
        self.grasp_in_progress = False
        self.box_attached = False

        self.moveit = None
        self.arm = None
        self.gripper = None
        self.gripper_action_client = None
        self.arm_action_client = None
        # grasp_timer and the gripper action client both need to run
        # concurrently: grasp_timer's callback blocks (via a
        # threading.Event, see _wait_for_future) waiting for the action
        # client's response callback to fire. Both landing in the node's
        # default MutuallyExclusiveCallbackGroup deadlocks -- confirmed in
        # isolated testing, the response callback can never run while
        # grasp_timer's callback (which is what's waiting for it) holds
        # that group's one execution slot. A shared ReentrantCallbackGroup
        # lets them interleave instead.
        self._reentrant_cbg = ReentrantCallbackGroup()

        # Only used to log real achieved-vs-commanded TCP accuracy under
        # physics actuation (see _log_arm_accuracy); harmless overhead in
        # fake-hardware mode where the snap is always exact anyway.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if MOVEIT_AVAILABLE:
            self._init_moveit()
        else:
            self.get_logger().warn("Running in mock mode, will log planned poses")

        if self.sim_gazebo:
            # MoveItPy's own controller-manager dispatch for the gripper
            # goes through trajectory_execution_manager's pre-execution
            # "is the current state fresh" check, which compares
            # MoveItPy's wall-clock time against /joint_states' Gazebo
            # sim-clock timestamps -- always stale, always ABORTED, gripper
            # goal never even sent (a confirmed, unresolved moveit_py
            # limitation: moveit/moveit2#2906). A plain gripper's open/
            # close doesn't need collision-aware motion planning anyway,
            # so bypass MoveIt entirely here and call the action server
            # directly -- this path was already proven to work in
            # isolated testing.
            self.gripper_action_client = ActionClient(
                self, GripperCommand, 'gripper_controller/gripper_cmd',
                callback_group=self._reentrant_cbg)
            # The arm's execute() calls turned out to hit the exact same
            # clock-mismatch abort, just inconsistently -- confirmed by
            # direct TF measurement: a "successful" (no exception, no
            # logged error) pre-grasp move left the real arm sitting at
            # home, never having moved at all, while the sequence carried
            # on as if it had. Same fix as the gripper: keep MoveIt for
            # planning (OMPL isn't clock-sensitive), bypass it for
            # execution by sending the planned trajectory to
            # ur_manipulator_controller directly.
            self.arm_action_client = ActionClient(
                self, FollowJointTrajectory,
                'ur_manipulator_controller/follow_joint_trajectory',
                callback_group=self._reentrant_cbg)

        self.pose_sub = self.create_subscription(
            PoseStamped, '/object_pose', self._pose_callback, 10
        )

        self.latest_joint_velocities = {}
        self.latest_joint_positions = {}
        self.latest_joint_efforts = {}
        if self.sim_gazebo:
            # gz_ros2_control's joint_state_broadcaster publishes with
            # best-effort QoS; a default (reliable) subscription here is
            # incompatible and silently never receives anything (bit us
            # once already, in gripper_effort_controller.py).
            from rclpy.qos import qos_profile_sensor_data
            self.create_subscription(
                JointState, '/joint_states', self._joint_state_cb,
                qos_profile_sensor_data)

        self.status_pub = self.create_publisher(String, '/grasp_status', 10)
        self.reinit_pub = self.create_publisher(Empty, '/foundationpose/reinit', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/grasp_target', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/grasp_markers', 10)
        self.attached_object_pub = self.create_publisher(
            AttachedCollisionObject, '/attached_collision_object', 10
        )

        self.grasp_timer = self.create_timer(
            1.0, self._grasp_timer_callback, callback_group=self._reentrant_cbg)

        self._publish_status(GraspState.IDLE)
        self.get_logger().info("Grasp executor ready. Waiting for stable pose...")

    def _init_moveit(self):
        """Initialize MoveIt2 using the generated ur5e_robotiq_moveit_config package."""
        try:
            self.get_logger().info("Initializing MoveIt2...")

            builder = MoveItConfigsBuilder("ur", package_name="ur5e_robotiq_moveit_config")
            if self.sim_gazebo:
                import os
                from ament_index_python.packages import get_package_share_directory
                pkg = get_package_share_directory('ur5e_robotiq_moveit_config')
                builder = builder.robot_description(mappings={
                    "sim_gazebo": "true",
                    "initial_positions_file": os.path.join(
                        pkg, 'config', 'initial_positions_gazebo.yaml'),
                })
            moveit_config = builder.moveit_cpp().planning_pipelines(pipelines=["ompl"]).to_moveit_configs()

            # gz_ros2_control stamps /joint_states with Gazebo's sim clock;
            # without MoveItPy's internal node also on sim time,
            # trajectory_execution_manager's "is this state fresh" check
            # compares two unrelated clocks and always fails. Harmless
            # (just a WARN) for the arm's FollowJointTrajectory controller,
            # but GripperCommand's handle treats it as fatal and never
            # sends the goal at all. Relies on this whole process having
            # been launched with `-p use_sim_time:=true` (ROS2's global
            # parameter override reaches this internal C++ node too);
            # injecting it directly into config_dict instead crashes with
            # an rclcpp qos_overrides exception.
            self.moveit = MoveItPy(
                node_name="grasp_executor_moveit",
                config_dict=moveit_config.to_dict(),
            )
            self.arm = self.moveit.get_planning_component(self.planning_group)
            self.gripper = self.moveit.get_planning_component(self.gripper_group)
            self.psm = self.moveit.get_planning_scene_monitor()
            self.get_logger().info("MoveIt2 initialized successfully!")
        except Exception as e:
            self.get_logger().error(f"MoveIt2 init failed: {e}")
            self.moveit = None
            self.arm = None
            self.gripper = None

    def _allow_gripper_box_collision(self, allowed: bool = True):
        """Allow the gripper's touch links to overlap sugar_box in the
        collision checker, since a real grasp always has them touching."""
        touch_links = [
            'robotiq_85_base_link',
            'robotiq_85_left_knuckle_link',
            'robotiq_85_right_knuckle_link',
            'robotiq_85_left_finger_link',
            'robotiq_85_right_finger_link',
            'robotiq_85_left_inner_knuckle_link',
            'robotiq_85_right_inner_knuckle_link',
            'robotiq_85_left_finger_tip_link',
            'robotiq_85_right_finger_tip_link',
        ]
        with self.psm.read_write() as scene:
            for link in touch_links:
                scene.allowed_collision_matrix.set_entry(link, 'sugar_box', allowed)
            # The perceived box height jitters by a few mm, so a box resting
            # on the table intermittently intersects it in the planning
            # model and fails the start-state collision check.
            scene.allowed_collision_matrix.set_entry('sugar_box', 'table', allowed)
            scene.current_state.update()

    def _joint_state_cb(self, msg: JointState):
        for name, vel in zip(msg.name, msg.velocity):
            self.latest_joint_velocities[name] = vel
        for name, pos in zip(msg.name, msg.position):
            self.latest_joint_positions[name] = pos
        for name, eff in zip(msg.name, msg.effort):
            self.latest_joint_efforts[name] = eff

    def _wait_for_arm_settle(self, timeout_sec: float = 8.0,
                              velocity_threshold: float = 0.01):
        """The controller reporting a trajectory finished only means it
        sent every commanded point in the trajectory's own (idealized,
        velocity-limit-based) time parameterization -- not that the real
        physics-actuated joint caught up to the last one. Confirmed by
        direct TF measurement: a "successful" grasp-descent move that
        returned in ~250ms left the real arm 12cm short of the commanded
        height. Poll real joint velocities until they're genuinely
        settled (or give up after timeout_sec) before trusting the
        position."""
        # A plain sleep, not rclpy.spin_once(): the joint_state
        # subscription callback (which updates latest_joint_velocities)
        # gets serviced by another MultiThreadedExecutor pool thread
        # regardless -- calling spin_once() here too raced against the
        # arm/gripper action clients' own wait-set management and
        # segfaulted rcl ("wait set index for feedback subscription is
        # out of bounds"), confirmed reproducible.
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            vels = self.latest_joint_velocities.values()
            if vels and all(abs(v) < velocity_threshold for v in vels):
                return True
            time.sleep(0.05)
        return False

    def _pose_callback(self, msg: PoseStamped):
        self.latest_pose = msg
        if self.collect_poses is not None and self._in_workspace(msg):
            self.collect_poses.append(msg)
        if self.grasp_in_progress:
            return
        self.pose_buffer.append(msg)
        if len(self.pose_buffer) > self.stability_count:
            self.pose_buffer.pop(0)

    def _is_pose_stable(self):
        if len(self.pose_buffer) < self.stability_count:
            return False
        positions = np.array([
            [p.pose.position.x, p.pose.position.y, p.pose.position.z]
            for p in self.pose_buffer
        ])
        std = np.std(positions, axis=0)
        max_std = np.max(std)
        if max_std < 0.02:
            self.get_logger().info(f"Pose stable! Std dev: {max_std*100:.1f}cm")
            return True
        return False

    def _compute_grasp_orientation(self, object_pose: Pose) -> Quaternion:
        """Compute a top-down grasp orientation with the gripper's jaw axis
        aligned to the box's short side, since the long side is close to
        the gripper's open width and would fail unpredictably."""
        box_quat_xyzw = [
            object_pose.orientation.x,
            object_pose.orientation.y,
            object_pose.orientation.z,
            object_pose.orientation.w,
        ]
        box_rot = Rotation.from_quat(box_quat_xyzw)

        # project the local X axis onto world XY to get yaw, robust to roll/pitch noise
        box_x_axis_world = box_rot.apply([1.0, 0.0, 0.0])
        box_yaw = np.arctan2(box_x_axis_world[1], box_x_axis_world[0])
        # The two-finger gripper is symmetric under a 180deg yaw, and a box's
        # pose estimate has that ambiguity anyway; fold into (-90, 90] so the
        # wrist always takes the reachable branch.
        if box_yaw > np.pi / 2:
            box_yaw -= np.pi
        elif box_yaw <= -np.pi / 2:
            box_yaw += np.pi

        r_point_down = Rotation.from_euler('x', 180, degrees=True)
        r_yaw = Rotation.from_euler('z', box_yaw)
        r_total = r_yaw * r_point_down  # apply point-down first, then yaw

        q = r_total.as_quat()
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    def _compute_grasp_pose(self, object_pose: PoseStamped) -> PoseStamped:
        """Compute the grasp target at the box's true geometric center."""
        center = get_box_center_from_bottom_pose(
            object_pose.pose, SUGAR_BOX_HALF_HEIGHT
        )

        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = object_pose.header.frame_id or 'world'
        grasp_pose.header.stamp = self.get_clock().now().to_msg()
        grasp_pose.pose.position = center.position
        grasp_pose.pose.orientation = self._compute_grasp_orientation(object_pose.pose)
        return grasp_pose

    def _compute_pre_grasp_pose(self, grasp_pose: PoseStamped) -> PoseStamped:
        pre_grasp = PoseStamped()
        pre_grasp.header = grasp_pose.header
        pre_grasp.pose.position.x = grasp_pose.pose.position.x
        pre_grasp.pose.position.y = grasp_pose.pose.position.y
        pre_grasp.pose.position.z = grasp_pose.pose.position.z + self.pre_grasp_height
        pre_grasp.pose.orientation = grasp_pose.pose.orientation
        return pre_grasp

    def _grasp_timer_callback(self):
        if self.grasp_in_progress:
            return
        if self.current_state == GraspState.IDLE:
            if self._is_pose_stable() and self.latest_pose is not None:
                self.get_logger().info("Stable pose detected! Starting grasp sequence...")
                self.grasp_in_progress = True
                self._execute_grasp_sequence(self.latest_pose)

    def _execute_grasp_sequence(self, object_pose: PoseStamped):
        try:
            grasp_pose = self._compute_grasp_pose(object_pose)
            pre_grasp_pose = self._compute_pre_grasp_pose(grasp_pose)

            self.target_pub.publish(grasp_pose)
            self._publish_grasp_markers(pre_grasp_pose, grasp_pose)

            self.get_logger().info(
                f"Grasp target: x={grasp_pose.pose.position.x:.3f}, "
                f"y={grasp_pose.pose.position.y:.3f}, "
                f"z={grasp_pose.pose.position.z:.3f}"
            )

            if self.arm is not None:
                success = self._moveit_execute_sequence(pre_grasp_pose, grasp_pose)
            else:
                success = self._mock_execute_sequence(pre_grasp_pose, grasp_pose)

            if success:
                self.current_state = GraspState.DONE
                self._publish_status(GraspState.DONE)
                self.get_logger().info("Grasp sequence completed successfully!")
            else:
                self._recover_if_attached()
                self.current_state = GraspState.ERROR
                self._publish_status(GraspState.ERROR)
                self.get_logger().error("Grasp sequence failed!")

        except Exception as e:
            self.get_logger().error(f"Grasp executor error: {e}")
            self._recover_if_attached()
            self.current_state = GraspState.ERROR
            self._publish_status(GraspState.ERROR)
        finally:
            self._reset_timer = self.create_timer(5.0, self._reset_once)

    # Above this, a settled arm is close enough; below the settle-wait's
    # velocity check can pass while the joint-space trajectory still left
    # real position error under the position-proportional velocity servo
    # (confirmed: one descent settled 42mm/7deg off target with velocities
    # already near zero). One corrective re-plan from the now-static
    # actual state converges easily since it's a tiny residual motion.
    _ARM_POSITION_TOLERANCE_M = 0.01

    def _plan_and_execute_arm(self, goal_pose: PoseStamped = None,
                               configuration_name: str = None,
                               _retry: bool = False) -> bool:
        self.arm.set_start_state_to_current_state()
        if configuration_name is not None:
            self.arm.set_goal_state(configuration_name=configuration_name)
        else:
            self.arm.set_goal_state(pose_stamped_msg=goal_pose, pose_link=self.eef_link)

        plan = self.arm.plan()
        if not plan and not _retry:
            # A stale sugar_box left in the planning scene (e.g. from
            # perception's first frames) can make the start state look
            # colliding for a moment; give the scene a moment to catch up.
            self.get_logger().warn("Planning failed; retrying once after 2 s")
            time.sleep(2.0)
            return self._plan_and_execute_arm(goal_pose, configuration_name, _retry=True)
        if not plan:
            return False

        if self.sim_gazebo and goal_pose is not None:
            self._log_planned_fk(plan, goal_pose)

        if self.sim_gazebo:
            joint_trajectory = plan.trajectory.get_robot_trajectory_msg().joint_trajectory
            if not self._send_arm_trajectory(joint_trajectory):
                return False
        else:
            self.moveit.execute(plan.trajectory, controllers=['ur_manipulator_controller'])

        if self.sim_gazebo and goal_pose is not None:
            pos_err = self._log_arm_accuracy(goal_pose)
            if (pos_err is not None
                    and pos_err > self._ARM_POSITION_TOLERANCE_M
                    and not _retry):
                self.get_logger().warn(
                    f"Arm settled {pos_err * 1000:.1f}mm off target; "
                    "issuing one corrective move"
                )
                # Best effort: the move itself already executed, so a failed
                # correction (e.g. the exact goal is in collision) is not fatal.
                if not self._plan_and_execute_arm(goal_pose=goal_pose, _retry=True):
                    self.get_logger().warn(
                        "Corrective move could not be planned; continuing from where the arm settled")
        return True

    def _send_arm_trajectory(self, joint_trajectory) -> bool:
        """Bypass MoveIt's execute() for the arm in physics mode -- same
        clock-mismatch abort as the gripper (see __init__), just
        inconsistent instead of always-fatal: confirmed by direct TF
        measurement that a "successful" execute() call had sometimes left
        the real arm at its start position, never having moved, while
        grasp_executor carried on as if the move had happened. OMPL
        planning above is untouched; only dispatch of the already-planned
        trajectory moves to a direct FollowJointTrajectory goal."""
        if not self.arm_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Arm action server not available")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_trajectory

        send_future = self.arm_action_client.send_goal_async(goal)
        self._wait_for_future(send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Arm trajectory goal rejected or timed out")
            return False

        result_future = goal_handle.get_result_async()
        # Generous timeout: trajectory duration varies with move distance.
        duration_sec = 0.0
        if joint_trajectory.points:
            d = joint_trajectory.points[-1].time_from_start
            duration_sec = d.sec + d.nanosec / 1e9
        self._wait_for_future(result_future, timeout_sec=duration_sec * 5.0 + 30.0)
        result = result_future.result()
        if result is None:
            self.get_logger().error("Arm trajectory timed out waiting for result")
            return False

        error_code = result.result.error_code
        if error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(f"Arm trajectory finished with error_code={error_code}")
            return False

        if not self._wait_for_arm_settle():
            self.get_logger().warn(
                "Arm joints still moving after settle timeout; proceeding anyway"
            )
        if joint_trajectory.points:
            tgt = dict(zip(joint_trajectory.joint_names, joint_trajectory.points[-1].positions))
            errs = {n: self.latest_joint_positions.get(n, float('nan')) - v for n, v in tgt.items()}
            self.get_logger().info(
                "JOINT ERR (rad): " + " ".join(f"{n.replace('_joint','')}={e:+.4f}" for n, e in errs.items())
                + " | EFFORT (Nm): " + " ".join(
                    f"{n.replace('_joint','')}={self.latest_joint_efforts.get(n, float('nan')):+.1f}"
                    for n in errs)
            )
        return True

    def _tcp_xyz(self, default):
        try:
            t = self.tf_buffer.lookup_transform(
                'world', self.eef_link, rclpy.time.Time(), Duration(seconds=0.5))
            p = t.transform.translation
            return (p.x, p.y, p.z)
        except Exception as e:
            self.get_logger().warn(f"TCP lookup failed ({e!r}); using commanded pose")
            return default

    def _log_arm_accuracy(self, goal_pose: PoseStamped):
        """Real achieved-vs-commanded TCP position right after execute()
        returns, under physics actuation. Diagnostic for grasp-pose
        positioning accuracy; a real gap here (not fake hardware's exact
        snap) is the leading suspect for the gripper closing on nothing
        instead of the box. Returns the position error in meters (None if
        the TF lookup failed) so callers can decide whether to correct."""
        try:
            t = self.tf_buffer.lookup_transform(
                goal_pose.header.frame_id or 'world', self.eef_link,
                rclpy.time.Time(), Duration(seconds=0.5))
            p = t.transform.translation
            q = t.transform.rotation
            gx = goal_pose.pose.position.x
            gy = goal_pose.pose.position.y
            gz = goal_pose.pose.position.z
            err = ((p.x - gx) ** 2 + (p.y - gy) ** 2 + (p.z - gz) ** 2) ** 0.5
            gq = goal_pose.pose.orientation
            # angle between achieved and commanded orientation, via quaternion dot product
            dot = abs(q.x * gq.x + q.y * gq.y + q.z * gq.z + q.w * gq.w)
            dot = min(1.0, dot)
            angle_err_deg = 2 * math.degrees(math.acos(dot))
            self.get_logger().info(
                f"ARM ACCURACY: commanded=({gx:.4f},{gy:.4f},{gz:.4f}) "
                f"achieved=({p.x:.4f},{p.y:.4f},{p.z:.4f}) "
                f"err=({p.x - gx:+.4f},{p.y - gy:+.4f},{p.z - gz:+.4f}) |err|={err:.4f} "
                f"orientation_err={angle_err_deg:.2f}deg "
                f"achieved_quat=({q.x:.4f},{q.y:.4f},{q.z:.4f},{q.w:.4f}) "
                f"commanded_quat=({gq.x:.4f},{gq.y:.4f},{gq.z:.4f},{gq.w:.4f})"
            )
            return err
        except Exception as e:
            self.get_logger().warn(f"ARM ACCURACY: TF lookup failed: {e!r}")
            return None

    # Must match the SRDF's gripper group_states (ur.srdf); MoveIt's own
    # plan-to-configuration_name path (used for fake hardware, below)
    # reads these from the SRDF directly, this direct-dispatch path can't.
    _GRIPPER_POSITIONS = {'open': 0.0, 'closed': 0.7929}

    def _plan_and_execute_gripper(self, configuration_name: str) -> bool:
        if self.sim_gazebo:
            return self._send_gripper_action(configuration_name)

        self.gripper.set_start_state_to_current_state()
        self.gripper.set_goal_state(configuration_name=configuration_name)
        plan = self.gripper.plan()
        if not plan:
            return False
        self.moveit.execute(plan.trajectory, controllers=['gripper_controller'])
        return True

    def _wait_for_future(self, future, timeout_sec: float):
        """Block the CALLING THREAD (not rclpy's spin machinery) until
        `future` completes. rclpy.spin_until_future_complete() calls
        executor.add_node()/remove_node() around the wait even when an
        executor is passed in explicitly -- fine for the default single-
        node, single-call use case, but from inside a callback that's
        already running on one thread of the very MultiThreadedExecutor
        spinning this node, that add/remove pair reliably deadlocked
        against the executor's own internal locking (confirmed: it hung
        indefinitely, ignoring timeout_sec, every single time). A plain
        threading.Event sidesteps rclpy's executor bookkeeping entirely:
        the future's own done-callback (fired from whichever pool thread
        services the action client's response) just sets the event, and
        this thread only ever blocks on that, never on the executor.
        """
        event = threading.Event()
        future.add_done_callback(lambda _f: event.set())
        event.wait(timeout=timeout_sec)

    def _send_gripper_action(self, configuration_name: str) -> bool:
        """Bypass MoveIt's controller-manager dispatch for the gripper in
        physics mode (see the comment in __init__ for why) and call the
        action server directly. No motion planning needed for a single-
        joint open/close, so nothing is lost by skipping it."""
        if not self.gripper_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper action server not available")
            return False

        goal = GripperCommand.Goal()
        goal.command.position = self._GRIPPER_POSITIONS[configuration_name]
        goal.command.max_effort = 10.0

        send_future = self.gripper_action_client.send_goal_async(goal)
        self._wait_for_future(send_future, timeout_sec=15.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected or timed out")
            return False

        result_future = goal_handle.get_result_async()
        self._wait_for_future(result_future, timeout_sec=30.0)
        result = result_future.result()
        if result is None:
            self.get_logger().error("Gripper action timed out waiting for result")
            return False

        r = result.result
        self.get_logger().info(
            f"Gripper {configuration_name}: position={r.position:.3f} "
            f"stalled={r.stalled}"
        )
        if configuration_name == 'closed' and not r.stalled:
            self.get_logger().error(
                "Gripper closed without stalling -- no contact detected, "
                "grasp missed the object"
            )
            return False
        return True

    def _attach_box(self):
        """Attach the sugar_box collision object to the gripper so lift
        planning treats it as part of the arm instead of an obstacle."""
        attached = AttachedCollisionObject()
        attached.link_name = 'robotiq_85_base_link'
        attached.object.header.frame_id = 'robotiq_85_base_link'
        attached.object.id = 'sugar_box'
        attached.object.operation = CollisionObject.ADD
        attached.touch_links = [
            'robotiq_85_base_link',
            'robotiq_85_left_knuckle_link',
            'robotiq_85_right_knuckle_link',
            'robotiq_85_left_finger_link',
            'robotiq_85_right_finger_link',
            'robotiq_85_left_inner_knuckle_link',
            'robotiq_85_right_inner_knuckle_link',
            'robotiq_85_left_finger_tip_link',
            'robotiq_85_right_finger_tip_link',
            'table',
        ]
        self.attached_object_pub.publish(attached)
        self.box_attached = True
        self.get_logger().info("Attach request sent for sugar_box")

    def _detach_box(self):
        """Detach the box and hand its collision object back to planning_scene_manager."""
        attached = AttachedCollisionObject()
        attached.object.id = 'sugar_box'
        attached.object.operation = CollisionObject.REMOVE
        self.attached_object_pub.publish(attached)
        self.box_attached = False
        self.get_logger().info("Detach request sent for sugar_box")

    def _recover_if_attached(self):
        """Detach the box before reporting a terminal error state.

        planning_scene_manager infers box_is_attached purely from
        /grasp_status (ATTACHED_STATES), not from whether _attach_box()
        actually fired. If a step after _attach_box() fails, the box
        stays attached in MoveIt's planning scene while ERROR flips
        planning_scene_manager back to republishing sugar_box as a
        world object, duplicating it. Detaching first keeps the two in
        sync.
        """
        if self.box_attached:
            self.get_logger().warn(
                "Sequence failed with the box still attached; detaching "
                "so it doesn't duplicate between world and attached."
            )
            self._detach_box()

    def _log_planned_fk(self, plan, goal_pose: PoseStamped):
        """Temporary root-cause diagnostic: FK the plan's own last waypoint
        BEFORE any execution happens, to tell whether a mismatch vs
        goal_pose originates in planning (IK/goal resolution) or in
        physics execution (servo/actuation)."""
        try:
            jt = plan.trajectory.get_robot_trajectory_msg().joint_trajectory
            if not jt.points:
                return
            last = jt.points[-1]
            with self.psm.read_only() as scene:
                robot_state = scene.current_state
                robot_state.set_joint_group_positions(
                    self.planning_group, list(last.positions)
                )
                robot_state.update()
                pose = robot_state.get_pose(self.eef_link)
                gx = goal_pose.pose.position.x
                gy = goal_pose.pose.position.y
                gz = goal_pose.pose.position.z
                err = ((pose.position.x - gx) ** 2 + (pose.position.y - gy) ** 2
                       + (pose.position.z - gz) ** 2) ** 0.5
                self.get_logger().info(
                    f"PLANNED-FK (pre-execution): goal=({gx:.4f},{gy:.4f},{gz:.4f}) "
                    f"plan_final_fk=({pose.position.x:.4f},{pose.position.y:.4f},{pose.position.z:.4f}) "
                    f"|err|={err:.4f}"
                )
        except Exception as e:
            self.get_logger().warn(f"PLANNED-FK diagnostic failed: {e!r}")

    def _diagnose_grasp_pose(self, grasp_pose: PoseStamped):
        """Log whether the grasp pose is reachable and collision-free.
        Uses read_only() so the IK check does not affect the live scene."""
        from moveit.core.collision_detection import CollisionRequest, CollisionResult

        with self.psm.read_only() as scene:
            robot_state = scene.current_state
            ik_success = robot_state.set_from_ik(
                self.planning_group, grasp_pose.pose, self.eef_link, 2.0
            )
            if not ik_success:
                self.get_logger().error(
                    "DIAGNOSTIC: IK failed for grasp pose, target is kinematically "
                    "unreachable at this position and orientation."
                )
                return

            robot_state.update()
            req = CollisionRequest()
            result = CollisionResult()
            scene.check_collision(req, result)

            if result.collision:
                self.get_logger().error(
                    "DIAGNOSTIC: IK succeeded but state is in collision."
                )
            else:
                self.get_logger().info(
                    "DIAGNOSTIC: IK succeeded and state is collision free, "
                    "the OMPL failure may be a sampling or tolerance issue."
                )

    @staticmethod
    def _in_workspace(msg: PoseStamped) -> bool:
        """Perception can emit garbage while initializing (and the box is
        high while carried); only trust poses of a box on the table."""
        p = msg.pose.position
        return 0.5 < p.x < 1.1 and -0.35 < p.y < 0.45 and 0.70 < p.z < 0.82

    def _moveit_execute_sequence(self, pre_grasp: PoseStamped, grasp: PoseStamped) -> bool:
        # grant this up front so the sequence recovers even if the robot starts in contact
        self._allow_gripper_box_collision(True)
        if self.mission:
            return self._run_mission(pre_grasp, grasp)
        return self._pick_and_place(pre_grasp, grasp, first_leg=True, target_xy=None)

    def _sample_place_target(self, avoid):
        """Random table spot at least min_place_separation from every point in `avoid`."""
        x0, x1 = self.get_parameter('place_x_min').value, self.get_parameter('place_x_max').value
        y0, y1 = self.get_parameter('place_y_min').value, self.get_parameter('place_y_max').value
        sep = self.get_parameter('min_place_separation').value
        for _ in range(200):
            x, y = float(self.rng.uniform(x0, x1)), float(self.rng.uniform(y0, y1))
            if all(((x - ax) ** 2 + (y - ay) ** 2) ** 0.5 >= sep for ax, ay in avoid):
                return x, y
        return x1, y1

    def _run_mission(self, pre1: PoseStamped, grasp1: PoseStamped) -> bool:
        """Leg 1: pick the box from home and put it down at a random spot.
        Leg 2: with the arm out of the way, SAM2 + FoundationPose find the box
        again from the camera alone; pick it from there and return it home."""
        home = (self.home_x, self.home_y)
        target1 = self._sample_place_target(avoid=[home])
        self.get_logger().info(
            f"MISSION leg 1: pick at ({grasp1.pose.position.x:.3f}, {grasp1.pose.position.y:.3f}), "
            f"place at random ({target1[0]:.3f}, {target1[1]:.3f})")
        if not self._pick_and_place(pre1, grasp1, first_leg=True, target_xy=target1,
                                    resample_avoid=[home]):
            return False

        found = self._redetect()
        if found is None:
            self.get_logger().error("MISSION: perception did not find the box after leg 1")
            return False
        self.get_logger().info(
            f"MISSION: perception found the box at ({found.pose.position.x:.3f}, "
            f"{found.pose.position.y:.3f}); target was ({self.leg1_target[0]:.3f}, "
            f"{self.leg1_target[1]:.3f})")

        grasp2 = self._compute_grasp_pose(found)
        pre2 = self._compute_pre_grasp_pose(grasp2)
        self.target_pub.publish(grasp2)
        self.get_logger().info(
            f"MISSION leg 2: pick at perceived ({grasp2.pose.position.x:.3f}, "
            f"{grasp2.pose.position.y:.3f}), return to home ({home[0]:.3f}, {home[1]:.3f})")
        if not self._pick_and_place(pre2, grasp2, first_leg=False, target_xy=home):
            return False

        final = self._redetect()
        if final is not None:
            d = ((final.pose.position.x - home[0]) ** 2 + (final.pose.position.y - home[1]) ** 2) ** 0.5
            self.get_logger().info(
                f"MISSION complete: perceived final box position ({final.pose.position.x:.3f}, "
                f"{final.pose.position.y:.3f}), {d * 1000:.0f} mm from home")
        return True

    def _redetect(self, timeout_sec: float = 60.0):
        """Ask FoundationPose to register the box afresh and return a stable,
        on-table pose from live perception (None on timeout)."""
        self._publish_status(GraspState.REDETECTING)
        self.get_logger().info("Re-detecting the box with live SAM2 + FoundationPose...")
        time.sleep(1.5)  # let the arm's motion settle out of view
        self.collect_poses = []
        self.reinit_pub.publish(Empty())
        deadline = time.monotonic() + timeout_sec
        try:
            while time.monotonic() < deadline:
                time.sleep(0.3)
                recent = list(self.collect_poses)[-10:]
                if len(recent) == 10:
                    arr = np.array([[m.pose.position.x, m.pose.position.y, m.pose.position.z]
                                    for m in recent])
                    if np.max(np.std(arr, axis=0)) < 0.01:
                        out = recent[-1]
                        out.pose.position.x, out.pose.position.y, out.pose.position.z = \
                            [float(v) for v in arr.mean(axis=0)]
                        return out
                if int(time.monotonic() - (deadline - timeout_sec)) % 15 == 14:
                    self.reinit_pub.publish(Empty())
        finally:
            self.collect_poses = None
        return None

    def _pick_and_place(self, pre_grasp: PoseStamped, grasp: PoseStamped,
                        first_leg: bool, target_xy, resample_avoid=None) -> bool:
        """Pick the box at `grasp`. With place enabled it is lifted fully,
        carried to target_xy (or pick + place_dx/dy when None), set down,
        released and the arm retreats home; otherwise it just lifts and
        returns home."""
        if first_leg:
            self._publish_status(GraspState.MOVING_HOME)
            self.get_logger().info("Moving to home position...")
            if not self._plan_and_execute_arm(configuration_name='home'):
                self.get_logger().error("Failed to plan to home position")
                return False

        self._publish_status(GraspState.OPENING_GRIPPER)
        self.get_logger().info("Opening gripper...")
        if not self._plan_and_execute_gripper('open'):
            self.get_logger().error("Failed to open gripper")
            return False

        self._publish_status(GraspState.PRE_GRASP)
        if not first_leg:
            # A direct home -> pre-grasp plan is unreliable toward the edges
            # of the table; going to a point high above the box first (the
            # same kind of pose the carry already reaches) and then straight
            # down is much more robust.
            high = self._shifted(pre_grasp, 0.0, 0.0,
                                 grasp.pose.position.z + 0.08 + self.lift_height)
            self.get_logger().info("Approaching from above...")
            if not self._plan_and_execute_arm(goal_pose=high):
                self.get_logger().error("Failed to plan approach above the box")
                return False
            if not self._straight_vertical(high, pre_grasp.pose.position.z, steps=6, duration_sec=4.0):
                self.get_logger().warn("Straight descent unavailable; planning the pre-grasp directly")
        self.get_logger().info("Moving to pre-grasp position...")
        if not self._plan_and_execute_arm(goal_pose=pre_grasp):
            self.get_logger().error("Failed to plan pre-grasp trajectory")
            return False

        self._publish_status(GraspState.GRASPING)
        self.get_logger().info("Moving down to grasp...")
        self._diagnose_grasp_pose(grasp)
        # A fast OMPL descent stalled ~8 cm short with the box in the way (the
        # same descent with no box, or in a much slower simulation, reached
        # full depth), so go down as a slow straight vertical line first.
        slow_ok = self.slow_descent and self._straight_vertical(
            pre_grasp, grasp.pose.position.z, steps=12, duration_sec=self.descent_duration)
        if slow_ok:
            self._log_arm_accuracy(grasp)
        elif not self._plan_and_execute_arm(goal_pose=grasp):
            self.get_logger().error("Failed to plan grasp trajectory")
            return False

        self._publish_status(GraspState.CLOSING_GRIPPER)
        self.get_logger().info("Closing gripper...")
        if not self._plan_and_execute_gripper('closed'):
            self.get_logger().error("Failed to close gripper")
            return False
        self._attach_box()
        gp = grasp.pose.position
        grasp_tcp = self._tcp_xyz(default=(gp.x, gp.y, gp.z))
        self.get_logger().info(
            f"GRASP TCP (achieved): ({grasp_tcp[0]:.3f}, {grasp_tcp[1]:.3f}, {grasp_tcp[2]:.3f}) "
            f"vs commanded ({gp.x:.3f}, {gp.y:.3f}, {gp.z:.3f})")

        self._publish_status(GraspState.LIFTING)
        if self.place_enabled:
            carry_z = grasp_tcp[2] + self.lift_height
            lift = self._shifted(grasp, 0.0, 0.0, carry_z)
            lift.pose.position.x, lift.pose.position.y = grasp_tcp[0], grasp_tcp[1]
            self.get_logger().info(f"Lifting fully to z={carry_z:.3f}...")
        else:
            lift = pre_grasp
            self.get_logger().info("Lifting...")
        if not self._plan_and_execute_arm(goal_pose=lift):
            self.get_logger().error("Failed to plan lift trajectory")
            return False

        if self.place_enabled:
            return self._place_sequence(grasp, grasp_tcp, target_xy, carry_z,
                                        verify=not self.mission,
                                        resample_avoid=resample_avoid)

        self._publish_status(GraspState.RETURNING_HOME)
        self.get_logger().info("Returning home...")
        if not self._plan_and_execute_arm(configuration_name='home'):
            self.get_logger().error("Failed to plan return-home trajectory")
            return False
        self._detach_box()
        self.get_logger().info("Object lifted and returned home!")
        return True

    _ARM_JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    def _straight_vertical(self, pose: PoseStamped, end_z: float, steps: int = 10,
                     duration_sec: float = 5.0) -> bool:
        """Move the TCP straight up from `pose` to end_z, keeping x, y and
        orientation, as a chain of IK waypoints sent as one slow trajectory.
        A free OMPL move here could swing sideways while the fingertips are
        still beside the box and drag or topple it."""
        try:
            positions = []
            with self.psm.read_only() as scene:
                rs = scene.current_state
                x, y = pose.pose.position.x, pose.pose.position.y
                z0 = pose.pose.position.z
                for i in range(1, steps + 1):
                    wp = Pose()
                    wp.position.x, wp.position.y = x, y
                    wp.position.z = z0 + (end_z - z0) * i / steps
                    wp.orientation = pose.pose.orientation
                    if not rs.set_from_ik(self.planning_group, wp, self.eef_link, 0.2):
                        self.get_logger().warn(f"Straight-up IK failed at step {i}")
                        return False
                    rs.update()
                    positions.append([float(v) for v in
                                      rs.get_joint_group_positions(self.planning_group)])
            from trajectory_msgs.msg import JointTrajectory
            jt = JointTrajectory()
            jt.joint_names = list(self._ARM_JOINTS)
            for i, q in enumerate(positions, start=1):
                pt = JointTrajectoryPoint()
                pt.positions = q
                pt.velocities = [0.0] * 6
                t = duration_sec * i / steps
                pt.time_from_start = DurationMsg(sec=int(t), nanosec=int((t % 1) * 1e9))
                jt.points.append(pt)
            return self._send_arm_trajectory(jt)
        except Exception as e:
            self.get_logger().warn(f"Straight-up retreat failed: {e!r}")
            return False

    def _shifted(self, pose: PoseStamped, dx: float, dy: float, z: float) -> PoseStamped:
        out = PoseStamped()
        out.header = pose.header
        out.pose.position.x = pose.pose.position.x + dx
        out.pose.position.y = pose.pose.position.y + dy
        out.pose.position.z = z
        out.pose.orientation = pose.pose.orientation
        return out

    def _place_sequence(self, grasp: PoseStamped, grasp_tcp, target_xy, carry_z: float,
                        verify: bool = True, resample_avoid=None) -> bool:
        """Carry the held box to target_xy, set it down, release, retreat.
        The place height is the TCP height actually reached when the
        gripper closed, so the box goes down to the table at the same grip
        offset it was picked up with."""
        gx, gy, gz = grasp_tcp
        if target_xy is None:
            target_xy = (gx + self.place_dx, gy + self.place_dy)
        tgt_x, tgt_y = target_xy

        self._publish_status(GraspState.TRANSPORTING)
        for attempt in range(3):
            pre_place = self._shifted(grasp, 0.0, 0.0, carry_z)
            pre_place.pose.position.x, pre_place.pose.position.y = tgt_x, tgt_y
            self.get_logger().info(
                f"Transporting to pre-place ({tgt_x:.3f}, {tgt_y:.3f}, {carry_z:.3f})...")
            if self._plan_and_execute_arm(goal_pose=pre_place):
                break
            if resample_avoid is None or attempt == 2:
                self.get_logger().error("Failed to plan transport trajectory")
                return False
            tgt_x, tgt_y = self._sample_place_target(avoid=resample_avoid)
            self.get_logger().warn(f"Target unreachable; trying another random spot ({tgt_x:.3f}, {tgt_y:.3f})")
        self.leg1_target = (tgt_x, tgt_y)

        self._publish_status(GraspState.PLACING)
        place = self._shifted(grasp, 0.0, 0.0, gz + 0.005)
        place.pose.position.x, place.pose.position.y = tgt_x, tgt_y
        self.get_logger().info(f"Lowering to place z={place.pose.position.z:.3f}...")
        if not self._plan_and_execute_arm(goal_pose=place):
            self.get_logger().error("Failed to plan place trajectory")
            return False

        self._detach_box()
        self._publish_status(GraspState.RELEASING)
        self.get_logger().info("Releasing...")
        if not self._plan_and_execute_gripper('open'):
            self.get_logger().error("Failed to open gripper at place")
            return False

        self._publish_status(GraspState.RETREATING)
        self.get_logger().info("Retreating straight up...")
        time.sleep(1.0)  # let the fingers finish opening before anything moves
        if not self._straight_vertical(place, carry_z):
            self.get_logger().warn("Straight retreat unavailable; falling back to a planned retreat")
            if not self._plan_and_execute_arm(goal_pose=pre_place):
                self.get_logger().error("Failed to plan retreat trajectory")
                return False
        if not self._plan_and_execute_arm(configuration_name='home'):
            self.get_logger().error("Failed to plan return-home trajectory")
            return False

        if verify:
            return self._verify_place(tgt_x, tgt_y)
        return True

    def _verify_place(self, tgt_x: float, tgt_y: float) -> bool:
        """With the arm out of the camera's way, wait for live SAM2 +
        FoundationPose to report a stable box pose and compare it to the
        intended place position."""
        self._publish_status(GraspState.VERIFYING_PLACE)
        self.get_logger().info("Verifying placement with live perception...")
        deadline = time.monotonic() + self.verify_timeout
        last_stamp = None
        recent = []
        while time.monotonic() < deadline:
            pose = self.latest_pose
            if pose is not None and (pose.header.stamp.sec, pose.header.stamp.nanosec) != last_stamp:
                last_stamp = (pose.header.stamp.sec, pose.header.stamp.nanosec)
                recent.append([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
                recent = recent[-10:]
                if len(recent) == 10 and np.max(np.std(np.array(recent), axis=0)) < 0.01:
                    m = np.mean(np.array(recent), axis=0)
                    err = ((m[0] - tgt_x) ** 2 + (m[1] - tgt_y) ** 2) ** 0.5
                    ok = err <= self.place_tolerance
                    self.place_result = {'perceived': m.tolist(), 'target': [tgt_x, tgt_y], 'err': err}
                    self.get_logger().info(
                        f"PLACE {'VERIFIED' if ok else 'OFF TARGET'}: perceived box at "
                        f"({m[0]:.3f}, {m[1]:.3f}), target ({tgt_x:.3f}, {tgt_y:.3f}), "
                        f"xy error {err * 1000:.1f} mm (tolerance {self.place_tolerance * 1000:.0f} mm)")
                    return ok
            time.sleep(0.2)
        self.get_logger().error("Placement not verified: no stable perceived pose before timeout")
        return False

    def _mock_execute_sequence(self, pre_grasp: PoseStamped, grasp: PoseStamped) -> bool:
        self.get_logger().info("=== MOCK GRASP SEQUENCE ===")
        self._publish_status(GraspState.MOVING_HOME)
        self.get_logger().info("Step 1: Move to HOME")
        self._publish_status(GraspState.OPENING_GRIPPER)
        self.get_logger().info("Step 2: OPEN GRIPPER")
        self._publish_status(GraspState.PRE_GRASP)
        self.get_logger().info(
            f"Step 3: PRE-GRASP, x={pre_grasp.pose.position.x:.3f}, "
            f"y={pre_grasp.pose.position.y:.3f}, z={pre_grasp.pose.position.z:.3f}"
        )
        self._publish_status(GraspState.GRASPING)
        self.get_logger().info(
            f"Step 4: GRASP, x={grasp.pose.position.x:.3f}, "
            f"y={grasp.pose.position.y:.3f}, z={grasp.pose.position.z:.3f}"
        )
        self._publish_status(GraspState.CLOSING_GRIPPER)
        self.get_logger().info("Step 5: CLOSE GRIPPER + ATTACH")
        self._publish_status(GraspState.LIFTING)
        self.get_logger().info(f"Step 6: LIFT, z={pre_grasp.pose.position.z:.3f}")
        self._publish_status(GraspState.RETURNING_HOME)
        self.get_logger().info("Step 7: RETURN HOME + DETACH")
        self.get_logger().info("=== MOCK GRASP COMPLETE ===")
        return True

    def _publish_grasp_markers(self, pre_grasp: PoseStamped, grasp: PoseStamped):
        markers = MarkerArray()
        pre_marker = Marker()
        pre_marker.header.frame_id = 'world'
        pre_marker.header.stamp = self.get_clock().now().to_msg()
        pre_marker.ns = 'grasp_poses'
        pre_marker.id = 0
        pre_marker.type = Marker.ARROW
        pre_marker.action = Marker.ADD
        pre_marker.pose = pre_grasp.pose
        pre_marker.scale.x = 0.12
        pre_marker.scale.y = 0.015
        pre_marker.scale.z = 0.015
        pre_marker.color.r = 0.0
        pre_marker.color.g = 0.0
        pre_marker.color.b = 1.0
        pre_marker.color.a = 0.8
        markers.markers.append(pre_marker)

        grasp_marker = Marker()
        grasp_marker.header.frame_id = 'world'
        grasp_marker.header.stamp = self.get_clock().now().to_msg()
        grasp_marker.ns = 'grasp_poses'
        grasp_marker.id = 1
        grasp_marker.type = Marker.ARROW
        grasp_marker.action = Marker.ADD
        grasp_marker.pose = grasp.pose
        grasp_marker.scale.x = 0.12
        grasp_marker.scale.y = 0.015
        grasp_marker.scale.z = 0.015
        grasp_marker.color.r = 0.0
        grasp_marker.color.g = 1.0
        grasp_marker.color.b = 0.0
        grasp_marker.color.a = 0.8
        markers.markers.append(grasp_marker)

        self.marker_pub.publish(markers)

    def _publish_status(self, state: str):
        self.current_state = state
        msg = String()
        msg.data = state
        self.status_pub.publish(msg)

    def _reset_once(self):
        if not self.single_shot:
            self._reset_state()
        self._reset_timer.cancel()

    def _reset_state(self):
        self.current_state = GraspState.IDLE
        self.grasp_in_progress = False
        self.pose_buffer.clear()
        self._publish_status(GraspState.IDLE)
        self.get_logger().info("Reset to IDLE, ready for next grasp")


def main(args=None):
    rclpy.init(args=args)
    node = GraspExecutorNode()
    # Single-threaded spin() can't service the gripper action client's
    # response callbacks while _send_gripper_action blocks waiting for
    # them from inside the grasp_timer callback -- that's a self-deadlock.
    # MultiThreadedExecutor lets another pool thread actually run them
    # while this one blocks (via a plain threading.Event in
    # _wait_for_future, not rclpy.spin_until_future_complete -- that
    # helper's own add_node()/remove_node() bookkeeping around the wait
    # deadlocks against this same executor's internal locking).
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()