#!/usr/bin/env python3
"""
Grasp Executor Node
-------------------
Subscribes to /object_pose from FoundationPose, computes a grasp
pose above the object, and uses MoveIt2 to plan and execute a
collision-free trajectory for the UR5e + Robotiq 2F-85.

Grasp strategy:
  1. Move to HOME (safe starting pose)
  2. Open gripper
  3. Move to PRE-GRASP (above the object, looking down)
  4. Move DOWN to GRASP, jaw axis aligned to the box's short side
  5. Close gripper, attach box collision object to the gripper
  6. LIFT (back to pre-grasp height)
  7. Return HOME, detach box collision object

Topics subscribed:
  /object_pose              (geometry_msgs/PoseStamped)   from FoundationPose

Topics published:
  /grasp_status              (std_msgs/String)             state machine state
  /grasp_target               (geometry_msgs/PoseStamped)   target grasp pose for RViz2
  /attached_collision_object  (moveit_msgs/AttachedCollisionObject)
"""

import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject

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
        # tcp_link, not tool0 — it already accounts for the adapter
        # plate + gripper body's physical reach past the wrist flange.
        self.declare_parameter('end_effector_link', 'tcp_link')
        self.declare_parameter('pre_grasp_height', 0.15)
        self.declare_parameter('pose_stability_count', 10)
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('max_velocity_scaling', 0.3)

        self.planning_group = self.get_parameter('planning_group').value
        self.gripper_group = self.get_parameter('gripper_group').value
        self.eef_link = self.get_parameter('end_effector_link').value
        self.pre_grasp_height = self.get_parameter('pre_grasp_height').value
        self.stability_count = self.get_parameter('pose_stability_count').value
        self.planning_time = self.get_parameter('planning_time').value
        self.velocity_scale = self.get_parameter('max_velocity_scaling').value

        self.get_logger().info("Grasp executor starting...")

        self.current_state = GraspState.IDLE
        self.latest_pose = None
        self.pose_buffer = []
        self.grasp_in_progress = False

        self.moveit = None
        self.arm = None
        self.gripper = None

        if MOVEIT_AVAILABLE:
            self._init_moveit()
        else:
            self.get_logger().warn("Running in mock mode — will log planned poses")

        self.pose_sub = self.create_subscription(
            PoseStamped, '/object_pose', self._pose_callback, 10
        )

        self.status_pub = self.create_publisher(String, '/grasp_status', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/grasp_target', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/grasp_markers', 10)
        self.attached_object_pub = self.create_publisher(
            AttachedCollisionObject, '/attached_collision_object', 10
        )

        self.grasp_timer = self.create_timer(1.0, self._grasp_timer_callback)

        self._publish_status(GraspState.IDLE)
        self.get_logger().info("Grasp executor ready. Waiting for stable pose...")

    def _init_moveit(self):
        """Initialize MoveIt2 Python bindings against the real generated
        MoveIt2 config package — not a hand-rolled approximation of it,
        so kinematics, controllers, and the gripper group are all loaded
        correctly rather than missing."""
        try:
            self.get_logger().info("Initializing MoveIt2...")

            moveit_config = MoveItConfigsBuilder(
                "ur", package_name="ur5e_robotiq_moveit_config"
            ).moveit_cpp().planning_pipelines(pipelines=["ompl"]).to_moveit_configs()

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
        """A real grasp target necessarily has the gripper body overlapping
        the object's collision volume — that's what grasping means — so
        OMPL reports zero valid states there unless we tell it this specific
        overlap is expected. Temporarily (and, once granted, permanently —
        the box becomes an attached object moments later anyway, so there's
        no meaningful moment to revoke this) allow the gripper's touch links
        to overlap sugar_box specifically, leaving every other collision
        check (table, arm, everything else) untouched."""
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
        """Top-down grasp, with the gripper's jaw axis (local X) aligned
        to the box's short axis (0.0495m), not its long axis (0.0942m,
        within 2mm of the gripper's 0.094m open width — approaching along
        that axis will fail unpredictably depending on the box's yaw)."""
        box_quat_xyzw = [
            object_pose.orientation.x,
            object_pose.orientation.y,
            object_pose.orientation.z,
            object_pose.orientation.w,
        ]
        box_rot = Rotation.from_quat(box_quat_xyzw)

        # Project the box's local X-axis onto the world XY plane to get
        # its yaw — more robust to small roll/pitch noise than pulling
        # yaw straight out of euler angles.
        box_x_axis_world = box_rot.apply([1.0, 0.0, 0.0])
        box_yaw = np.arctan2(box_x_axis_world[1], box_x_axis_world[0])

        r_point_down = Rotation.from_euler('x', 180, degrees=True)
        r_yaw = Rotation.from_euler('z', box_yaw)
        r_total = r_yaw * r_point_down  # apply point-down first, then yaw

        q = r_total.as_quat()
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    def _compute_grasp_pose(self, object_pose: PoseStamped) -> PoseStamped:
        """Grasp target = box's true geometric center. No extra flange
        offset needed here — tcp_link already encodes that in the URDF."""
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
                self.current_state = GraspState.ERROR
                self._publish_status(GraspState.ERROR)
                self.get_logger().error("Grasp sequence failed!")

        except Exception as e:
            self.get_logger().error(f"Grasp executor error: {e}")
            self.current_state = GraspState.ERROR
            self._publish_status(GraspState.ERROR)
        finally:
            self._reset_timer = self.create_timer(5.0, self._reset_once)

    def _plan_and_execute_arm(self, goal_pose: PoseStamped = None,
                               configuration_name: str = None) -> bool:
        self.arm.set_start_state_to_current_state()
        if configuration_name is not None:
            self.arm.set_goal_state(configuration_name=configuration_name)
        else:
            self.arm.set_goal_state(pose_stamped_msg=goal_pose, pose_link=self.eef_link)

        plan = self.arm.plan()
        if not plan:
            return False
        self.moveit.execute(plan.trajectory, controllers=[])
        return True

    def _plan_and_execute_gripper(self, configuration_name: str) -> bool:
        self.gripper.set_start_state_to_current_state()
        self.gripper.set_goal_state(configuration_name=configuration_name)
        plan = self.gripper.plan()
        if not plan:
            return False
        self.moveit.execute(plan.trajectory, controllers=[])
        return True

    def _attach_box(self):
        """Move the sugar_box collision object from world to attached-on-
        gripper, so lift planning treats it as part of the arm rather
        than a static obstacle it's now holding. NOTE: attach/detach
        semantics here are implemented per the documented
        AttachedCollisionObject message pattern but not yet verified
        against real behavior in this workspace — check RViz's Scene
        Objects tab after this fires to confirm sugar_box actually moves
        from the world list to attached, rather than duplicating."""
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
        self.get_logger().info("Attach request sent for sugar_box")

    def _detach_box(self):
        """Detach — planning_scene_manager will keep republishing the
        box's world collision object at its live /object_pose-derived
        position afterward anyway, so this just hands control back."""
        attached = AttachedCollisionObject()
        attached.object.id = 'sugar_box'
        attached.object.operation = CollisionObject.REMOVE
        self.attached_object_pub.publish(attached)
        self.get_logger().info("Detach request sent for sugar_box")

    def _diagnose_grasp_pose(self, grasp_pose: PoseStamped):
        """Read-only diagnostic. Uses read_only(), not read_write() — the
        previous version used read_write() and its IK solution got
        committed back into the live shared robot state on exit, corrupting
        every subsequent plan (that's what caused 'home' to suddenly show
        a base_link/table collision on the very next attempt). read_only()
        should not persist anything on exit, so this version is safe to
        call without side effects. Also drops the earlier .contacts access,
        which isn't convertible to a Python type on this pybind11 build and
        crashed the node outright — a plain collision boolean is enough to
        answer the actual question."""
        from moveit.core.collision_detection import CollisionRequest, CollisionResult

        with self.psm.read_only() as scene:
            robot_state = scene.current_state
            ik_success = robot_state.set_from_ik(
                self.planning_group, grasp_pose.pose, self.eef_link, 2.0
            )
            if not ik_success:
                self.get_logger().error(
                    "DIAGNOSTIC: IK FAILED for grasp pose — target is kinematically "
                    "unreachable at this position/orientation, not a collision issue."
                )
                return

            robot_state.update()
            req = CollisionRequest()
            result = CollisionResult()
            scene.check_collision(req, result)

            if result.collision:
                self.get_logger().error(
                    "DIAGNOSTIC: IK succeeded but state IS in collision."
                )
            else:
                self.get_logger().info(
                    "DIAGNOSTIC: IK succeeded and state is collision-free — "
                    "the OMPL failure may be a sampling/tolerance issue, not "
                    "reachability or collision."
                )

    def _moveit_execute_sequence(self, pre_grasp: PoseStamped, grasp: PoseStamped) -> bool:
        # Grant this up front, not just before the descent — makes the
        # sequence recoverable even if the robot starts already in contact
        # with the box (e.g. left over from an interrupted previous run).
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
            f"Step 3: PRE-GRASP → x={pre_grasp.pose.position.x:.3f}, "
            f"y={pre_grasp.pose.position.y:.3f}, z={pre_grasp.pose.position.z:.3f}"
        )
        self._publish_status(GraspState.GRASPING)
        self.get_logger().info(
            f"Step 4: GRASP → x={grasp.pose.position.x:.3f}, "
            f"y={grasp.pose.position.y:.3f}, z={grasp.pose.position.z:.3f}"
        )
        self._publish_status(GraspState.CLOSING_GRIPPER)
        self.get_logger().info("Step 5: CLOSE GRIPPER + ATTACH")
        self._publish_status(GraspState.LIFTING)
        self.get_logger().info(f"Step 6: LIFT → z={pre_grasp.pose.position.z:.3f}")
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
        self._reset_state()
        self._reset_timer.cancel()

    def _reset_state(self):
        self.current_state = GraspState.IDLE
        self.grasp_in_progress = False
        self.pose_buffer.clear()
        self._publish_status(GraspState.IDLE)
        self.get_logger().info("Reset to IDLE — ready for next grasp")


def main(args=None):
    rclpy.init(args=args)
    node = GraspExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()