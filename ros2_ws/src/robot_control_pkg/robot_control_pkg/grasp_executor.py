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
from std_msgs.msg import String
from sensor_msgs.msg import JointState
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

        self.planning_group = self.get_parameter('planning_group').value
        self.gripper_group = self.get_parameter('gripper_group').value
        self.eef_link = self.get_parameter('end_effector_link').value
        self.pre_grasp_height = self.get_parameter('pre_grasp_height').value
        self.stability_count = self.get_parameter('pose_stability_count').value
        self.planning_time = self.get_parameter('planning_time').value
        self.velocity_scale = self.get_parameter('max_velocity_scaling').value
        self.sim_gazebo = self.get_parameter('sim_gazebo').value
        self.single_shot = self.get_parameter('single_shot').value

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
            scene.current_state.update()

    def _joint_state_cb(self, msg: JointState):
        for name, vel in zip(msg.name, msg.velocity):
            self.latest_joint_velocities[name] = vel
        for name, pos in zip(msg.name, msg.position):
            self.latest_joint_positions[name] = pos

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
                return self._plan_and_execute_arm(goal_pose=goal_pose, _retry=True)
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
            )
        return True

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

    def _moveit_execute_sequence(self, pre_grasp: PoseStamped, grasp: PoseStamped) -> bool:
        # grant this up front so the sequence recovers even if the robot starts in contact
        self._allow_gripper_box_collision(True)

        # Step 1: home
        self._publish_status(GraspState.MOVING_HOME)
        self.get_logger().info("Moving to home position...")
        if not self._plan_and_execute_arm(configuration_name='home'):
            self.get_logger().error("Failed to plan to home position")
            return False

        # Step 2: open gripper before descending
        self._publish_status(GraspState.OPENING_GRIPPER)
        self.get_logger().info("Opening gripper...")
        if not self._plan_and_execute_gripper('open'):
            self.get_logger().error("Failed to open gripper")
            return False

        # Step 3: pre-grasp
        self._publish_status(GraspState.PRE_GRASP)
        self.get_logger().info("Moving to pre-grasp position...")
        if not self._plan_and_execute_arm(goal_pose=pre_grasp):
            self.get_logger().error("Failed to plan pre-grasp trajectory")
            return False

        # Step 4: down to grasp
        self._publish_status(GraspState.GRASPING)
        self.get_logger().info("Moving down to grasp...")
        self._diagnose_grasp_pose(grasp)
        if not self._plan_and_execute_arm(goal_pose=grasp):
            self.get_logger().error("Failed to plan grasp trajectory")
            return False

        # Step 5: close gripper + attach
        self._publish_status(GraspState.CLOSING_GRIPPER)
        self.get_logger().info("Closing gripper...")
        if not self._plan_and_execute_gripper('closed'):
            self.get_logger().error("Failed to close gripper")
            return False
        self._attach_box()

        # Step 6: lift
        self._publish_status(GraspState.LIFTING)
        self.get_logger().info("Lifting...")
        if not self._plan_and_execute_arm(goal_pose=pre_grasp):
            self.get_logger().error("Failed to plan lift trajectory")
            return False

        # Step 7: return home + detach
        self._publish_status(GraspState.RETURNING_HOME)
        self.get_logger().info("Returning home...")
        if not self._plan_and_execute_arm(configuration_name='home'):
            self.get_logger().error("Failed to plan return-home trajectory")
            return False
        self._detach_box()

        self.get_logger().info("Object lifted and returned home!")
        return True

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