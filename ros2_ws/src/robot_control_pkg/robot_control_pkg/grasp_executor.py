#!/usr/bin/env python3
"""
Grasp Executor Node
-------------------
Subscribes to /object_pose from FoundationPose, computes a grasp
pose above the object, and uses MoveIt2 to plan and execute a
collision-free trajectory for the UR5e robot arm.

The grasp strategy is simple but effective:
  1. Move to a HOME position (safe starting pose)
  2. Move to PRE-GRASP position (above the object, looking down)
  3. Move DOWN to GRASP position (at the object)
  4. Close gripper (simulated)
  5. Move back UP to POST-GRASP (lift the object)
  6. Return HOME

Topics subscribed:
  /object_pose    (geometry_msgs/PoseStamped)  from FoundationPose

Topics published:
  /grasp_status   (std_msgs/String)  current state machine state
  /grasp_target   (geometry_msgs/PoseStamped)  target grasp pose for RViz2
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import numpy as np

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

# MoveIt2 Python bindings
try:
    from moveit.planning import MoveItPy
    from moveit.core.robot_state import RobotState
    MOVEIT_AVAILABLE = True
except ImportError:
    print("[WARN] MoveIt2 Python bindings not available, using mock mode")
    MOVEIT_AVAILABLE = False

from scipy.spatial.transform import Rotation


# State machine for the grasp sequence
class GraspState:
    IDLE = "IDLE"                   # waiting for a pose
    MOVING_HOME = "MOVING_HOME"     # moving to safe home position
    PRE_GRASP = "PRE_GRASP"         # moving above the object
    GRASPING = "GRASPING"           # moving down to the object
    LIFTING = "LIFTING"             # lifting the object up
    DONE = "DONE"                   # grasp complete
    ERROR = "ERROR"                 # something went wrong


class GraspExecutorNode(Node):
    """
    MoveIt2-based grasp executor for the FoundationPose pipeline.

    Receives object poses from FoundationPose, computes grasp poses,
    and executes the full pick sequence using MoveIt2 motion planning.
    """

    def __init__(self):
        super().__init__('grasp_executor')

        self.declare_parameter('robot_name', 'ur5e')
        self.declare_parameter('planning_group', 'ur_manipulator')
        self.declare_parameter('end_effector_link', 'tool0')
        self.declare_parameter('pre_grasp_height', 0.15)
        # Wait for 10 stable poses to filter out noise
        self.declare_parameter('pose_stability_count', 10)
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('max_velocity_scaling', 0.3)

        self.robot_name = self.get_parameter('robot_name').value
        self.planning_group = self.get_parameter('planning_group').value
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

        if MOVEIT_AVAILABLE:
            self._init_moveit()
        else:
            self.get_logger().warn("Running in mock mode — will log planned poses")

        # Wait for stable poses before triggering a grasp
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/object_pose',
            self._pose_callback,
            10
        )

        self.status_pub = self.create_publisher(String, '/grasp_status', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/grasp_target', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/grasp_markers', 10)

        # Check every second if we have a stable pose to grasp
        self.grasp_timer = self.create_timer(1.0, self._grasp_timer_callback)

        self._publish_status(GraspState.IDLE)
        self.get_logger().info("Grasp executor ready. Waiting for stable pose...")

    def _init_moveit(self):
        """Initialize MoveIt2 Python bindings and get the planning component."""
        try:
            self.get_logger().info("Initializing MoveIt2...")
            self.moveit = MoveItPy(node_name="grasp_executor_moveit")
            self.arm = self.moveit.get_planning_component(self.planning_group)
            self.get_logger().info("MoveIt2 initialized successfully!")
        except Exception as e:
            self.get_logger().error(f"MoveIt2 init failed: {e}")
            self.moveit = None
            self.arm = None

    def _pose_callback(self, msg: PoseStamped):
        """Buffer incoming poses to check for stability before acting."""
        self.latest_pose = msg

        if self.grasp_in_progress:
            return

        self.pose_buffer.append(msg)

        if len(self.pose_buffer) > self.stability_count:
            self.pose_buffer.pop(0)

    def _is_pose_stable(self):
        """
        Check if buffered poses are consistent enough to grasp.
        Stable if position std dev < 2cm across all axes.
        """
        if len(self.pose_buffer) < self.stability_count:
            return False

        positions = np.array([
            [p.pose.position.x,
             p.pose.position.y,
             p.pose.position.z]
            for p in self.pose_buffer
        ])

        std = np.std(positions, axis=0)
        max_std = np.max(std)

        is_stable = max_std < 0.02

        if is_stable:
            self.get_logger().info(
                f"Pose stable! Std dev: {max_std*100:.1f}cm"
            )
        return is_stable

    def _compute_grasp_pose(self, object_pose: PoseStamped) -> PoseStamped:
        """
        Compute the grasp pose. 
        Approaches from above with the gripper pointing straight down.
        The 0.16m Z offset accounts for the UR5e tool flange to finger tip.
        """
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = 'world'
        grasp_pose.header.stamp = self.get_clock().now().to_msg()

        grasp_pose.pose.position.x = object_pose.pose.position.x
        grasp_pose.pose.position.y = object_pose.pose.position.y
        # Z offset so gripper fingers reach the object center
        grasp_pose.pose.position.z = object_pose.pose.position.z + 0.16

        # Rotate 180deg around X to point the tool straight down
        r = Rotation.from_euler('xyz', [180, 0, 0], degrees=True)
        quat = r.as_quat()  # [x, y, z, w]

        grasp_pose.pose.orientation.x = quat[0]
        grasp_pose.pose.orientation.y = quat[1]
        grasp_pose.pose.orientation.z = quat[2]
        grasp_pose.pose.orientation.w = quat[3]

        return grasp_pose

    def _compute_pre_grasp_pose(self, grasp_pose: PoseStamped) -> PoseStamped:
        """Compute the pre-grasp pose directly above the grasp pose for a straight vertical approach."""
        pre_grasp = PoseStamped()
        pre_grasp.header = grasp_pose.header

        pre_grasp.pose.position.x = grasp_pose.pose.position.x
        pre_grasp.pose.position.y = grasp_pose.pose.position.y
        pre_grasp.pose.position.z = (
            grasp_pose.pose.position.z + self.pre_grasp_height
        )
        pre_grasp.pose.orientation = grasp_pose.pose.orientation

        return pre_grasp

    def _grasp_timer_callback(self):
        """Check if we should trigger a grasp sequence."""
        if self.grasp_in_progress:
            return

        if self.current_state == GraspState.IDLE:
            if self._is_pose_stable() and self.latest_pose is not None:
                self.get_logger().info(
                    "Stable pose detected! Starting grasp sequence..."
                )
                self.grasp_in_progress = True
                self._execute_grasp_sequence(self.latest_pose)

    def _execute_grasp_sequence(self, object_pose: PoseStamped):
        """Execute the full pick sequence: HOME -> PRE_GRASP -> GRASP -> LIFT -> HOME."""
        try:
            grasp_pose = self._compute_grasp_pose(object_pose)
            pre_grasp_pose = self._compute_pre_grasp_pose(grasp_pose)

            self.target_pub.publish(grasp_pose)
            self._publish_grasp_markers(pre_grasp_pose, grasp_pose)

            self.get_logger().info(
                f"Grasp target: "
                f"x={grasp_pose.pose.position.x:.3f}, "
                f"y={grasp_pose.pose.position.y:.3f}, "
                f"z={grasp_pose.pose.position.z:.3f}"
            )

            if self.arm is not None:
                success = self._moveit_execute_sequence(
                    pre_grasp_pose, grasp_pose
                )
            else:
                success = self._mock_execute_sequence(
                    pre_grasp_pose, grasp_pose
                )

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

    def _moveit_execute_sequence(self, pre_grasp: PoseStamped, grasp: PoseStamped) -> bool:
        """Execute the grasp sequence using real MoveIt2 planning."""
        robot_model = self.moveit.get_robot_model()
        robot_state = RobotState(robot_model)

        # Step 1: Move to home (predefined safe configuration in SRDF)
        self._publish_status(GraspState.MOVING_HOME)
        self.get_logger().info("Moving to home position...")

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name='home')

        plan = self.arm.plan()
        if not plan:
            self.get_logger().error("Failed to plan to home position")
            return False

        self.moveit.execute(plan.trajectory, controllers=[])
        self.get_logger().info("At home position")

        # Step 2: Move to pre-grasp
        self._publish_status(GraspState.PRE_GRASP)
        self.get_logger().info("Moving to pre-grasp position...")

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=pre_grasp, pose_link=self.eef_link)

        plan = self.arm.plan()
        if not plan:
            self.get_logger().error("Failed to plan pre-grasp trajectory")
            return False

        self.moveit.execute(plan.trajectory, controllers=[])
        self.get_logger().info("At pre-grasp position")

        # Step 3: Move down to grasp
        self._publish_status(GraspState.GRASPING)
        self.get_logger().info("Moving down to grasp...")

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=grasp, pose_link=self.eef_link)

        plan = self.arm.plan()
        if not plan:
            self.get_logger().error("Failed to plan grasp trajectory")
            return False

        self.moveit.execute(plan.trajectory, controllers=[])
        self.get_logger().info("At grasp position — closing gripper")

        # Step 4: Gripper close (simulated)
        self.get_logger().info("Gripper closed (simulated)")

        # Step 5: Lift up
        self._publish_status(GraspState.LIFTING)

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=pre_grasp, pose_link=self.eef_link)

        plan = self.arm.plan()
        if not plan:
            self.get_logger().error("Failed to plan lift trajectory")
            return False

        self.moveit.execute(plan.trajectory, controllers=[])
        self.get_logger().info("Object lifted!")

        return True

    def _mock_execute_sequence(self, pre_grasp: PoseStamped, grasp: PoseStamped) -> bool:
        """Log the poses that would be executed if MoveIt2 were available."""
        self.get_logger().info("=== MOCK GRASP SEQUENCE ===")

        self._publish_status(GraspState.MOVING_HOME)
        self.get_logger().info("Step 1: Move to HOME")

        self._publish_status(GraspState.PRE_GRASP)
        self.get_logger().info(
            f"Step 2: PRE-GRASP → "
            f"x={pre_grasp.pose.position.x:.3f}, "
            f"y={pre_grasp.pose.position.y:.3f}, "
            f"z={pre_grasp.pose.position.z:.3f}"
        )

        self._publish_status(GraspState.GRASPING)
        self.get_logger().info(
            f"Step 3: GRASP → "
            f"x={grasp.pose.position.x:.3f}, "
            f"y={grasp.pose.position.y:.3f}, "
            f"z={grasp.pose.position.z:.3f}"
        )

        self.get_logger().info("Step 4: CLOSE GRIPPER")

        self._publish_status(GraspState.LIFTING)
        self.get_logger().info(
            f"Step 5: LIFT → "
            f"z={pre_grasp.pose.position.z:.3f}"
        )

        self.get_logger().info("=== MOCK GRASP COMPLETE ===")
        return True

    def _publish_grasp_markers(self, pre_grasp: PoseStamped, grasp: PoseStamped):
        """Publish RViz2 markers (blue for pre-grasp, green for grasp) for visual debugging."""
        markers = MarkerArray()

        # Pre-grasp marker (blue)
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

        # Grasp marker (green)
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
        """Publish current state machine state."""
        self.current_state = state
        msg = String()
        msg.data = state
        self.status_pub.publish(msg)

    def _reset_once(self):
        self._reset_state()
        self._reset_timer.cancel()

    def _reset_state(self):
        """Reset back to IDLE after a grasp attempt."""
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