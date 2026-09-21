#!/usr/bin/env python3
"""Closed-loop, effort-based gripper control for the physics-mode
(gz_ros2_control) manipulation stack.

position_controllers/GripperActionController drives the knuckle joint to a
fixed POSITION goal. Under real physics that turned out not to reliably
translate into real squeeze force: gz_ros2_control's position command
interface is a velocity servo (joint_velocity = gain * error * rate), and
tuning that gain for genuine contact force against an obstruction (as
opposed to just holding a pose against gravity) proved unreliable in
testing -- MoveIt reported the gripper goal reached while the box itself
never left the table.

This node instead commands a constant TORQUE (through a plain
forward_command_controller/ForwardCommandController that just forwards
whatever effort value it's given) and watches real /joint_states feedback
to detect contact: apply a modest closing effort, and if the joint stalls
(near-zero velocity for a sustained window) before reaching the fully
open (or closed) position, that's real physical contact, not a spring
holding a pose. It reports that back as `stalled` in the standard
control_msgs/action/GripperCommand result -- a genuine sensor-driven
grasp-success signal, not "the trajectory executed".

Runs as a plain rclpy node (not a ros2_control controller) presenting the
same action name/type MoveIt already expects
(<gripper_controller>/gripper_cmd, control_msgs/action/GripperCommand),
so grasp_executor needs no changes to use it.
"""
import time

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class GripperEffortController(Node):
    def __init__(self):
        super().__init__('gripper_effort_controller')

        self.declare_parameter('joint_name', 'robotiq_85_left_knuckle_joint')
        self.declare_parameter('command_topic', '/gripper_controller/commands')
        self.declare_parameter('action_name', 'gripper_controller/gripper_cmd')
        self.declare_parameter('open_position', 0.0)
        self.declare_parameter('closed_position', 0.7929)
        self.declare_parameter('position_tolerance', 0.015)
        self.declare_parameter('close_effort', 6.0)
        self.declare_parameter('open_effort', -4.0)
        self.declare_parameter('hold_effort', 8.0)
        self.declare_parameter('rest_open_effort', -1.0)
        self.declare_parameter('stall_velocity_threshold', 0.01)
        self.declare_parameter('stall_dwell_sec', 0.3)
        self.declare_parameter('motion_timeout_sec', 12.0)
        self.declare_parameter('control_period_sec', 0.02)

        self.joint_name = self.get_parameter('joint_name').value
        self.open_position = self.get_parameter('open_position').value
        self.closed_position = self.get_parameter('closed_position').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.close_effort = self.get_parameter('close_effort').value
        self.open_effort = self.get_parameter('open_effort').value
        self.hold_effort = self.get_parameter('hold_effort').value
        self.rest_open_effort = self.get_parameter('rest_open_effort').value
        self.stall_velocity_threshold = self.get_parameter('stall_velocity_threshold').value
        self.stall_dwell_sec = self.get_parameter('stall_dwell_sec').value
        self.motion_timeout_sec = self.get_parameter('motion_timeout_sec').value
        self.control_period = self.get_parameter('control_period_sec').value

        self.latest_position = None
        self.latest_velocity = None
        self.latest_effort = None

        # joint_state_broadcaster publishes with best-effort QoS; a
        # default (reliable) subscription here is incompatible and
        # silently never receives anything.
        self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, qos_profile_sensor_data)
        self.effort_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter('command_topic').value, 10)

        self._commanded_effort = self.rest_open_effort
        self.create_timer(self.control_period, self._publish_effort)

        self._action_server = ActionServer(
            self, GripperCommand, self.get_parameter('action_name').value,
            self._execute_callback, callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(
            f"Gripper effort controller ready: close={self.close_effort} Nm, "
            f"hold={self.hold_effort} Nm, open={self.open_effort} Nm"
        )

    def _joint_state_cb(self, msg: JointState):
        try:
            idx = msg.name.index(self.joint_name)
        except ValueError:
            return
        self.latest_position = msg.position[idx]
        if msg.velocity and len(msg.velocity) > idx:
            self.latest_velocity = msg.velocity[idx]
        if msg.effort and len(msg.effort) > idx:
            self.latest_effort = msg.effort[idx]

    def _publish_effort(self):
        msg = Float64MultiArray()
        msg.data = [self._commanded_effort, -self._commanded_effort]
        self.effort_pub.publish(msg)

    def _wait_for_joint_state(self, timeout_sec=5.0):
        deadline = time.monotonic() + timeout_sec
        while self.latest_position is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.latest_position is not None

    async def _execute_callback(self, goal_handle):
        goal = goal_handle.request.command
        closing = goal.position > (self.open_position + self.closed_position) / 2.0

        if not self._wait_for_joint_state():
            self.get_logger().error("No /joint_states received for gripper joint")
            goal_handle.abort()
            return GripperCommand.Result()

        target_position = self.closed_position if closing else self.open_position
        driving_effort = self.close_effort if closing else self.open_effort
        direction = "closing" if closing else "opening"
        self.get_logger().info(
            f"Gripper {direction}: driving at {driving_effort} Nm toward {target_position:.3f}"
        )

        self._commanded_effort = driving_effort
        deadline = time.monotonic() + self.motion_timeout_sec
        stall_since = None
        stalled = False

        while time.monotonic() < deadline:
            time.sleep(self.control_period)
            pos = self.latest_position
            vel = self.latest_velocity if self.latest_velocity is not None else 0.0

            feedback = GripperCommand.Feedback()
            feedback.position = pos
            feedback.effort = self.latest_effort if self.latest_effort is not None else 0.0
            feedback.stalled = False
            feedback.reached_goal = False
            goal_handle.publish_feedback(feedback)

            reached_target = abs(pos - target_position) <= self.position_tolerance
            if reached_target:
                break

            if abs(vel) < self.stall_velocity_threshold:
                if stall_since is None:
                    stall_since = time.monotonic()
                elif time.monotonic() - stall_since >= self.stall_dwell_sec:
                    stalled = True
                    break
            else:
                stall_since = None

        final_position = self.latest_position if self.latest_position is not None else 0.0
        final_effort = self.latest_effort if self.latest_effort is not None else 0.0

        if closing and not stalled and abs(final_position - target_position) > self.position_tolerance:
            # Timed out well short of the fully-closed target without a
            # clean continuous stall dwell. Real contact often makes the
            # velocity signal jittery (small bounces/slip against the
            # object keep nudging it back above stall_velocity_threshold),
            # repeatedly resetting the dwell timer before it ever
            # accumulates stall_dwell_sec -- so "never cleanly stalled but
            # also never reached the fully-closed position" is itself
            # evidence of an obstruction, not free motion.
            stalled = True
            self.get_logger().info(
                f"Gripper stopped short of target ({final_position:.3f} rad vs "
                f"{target_position:.3f} rad) without a clean stall dwell -- "
                "treating as contact (noisy velocity signal)."
            )

        if closing and stalled:
            # Real contact before reaching the fully-closed position: an
            # object is between the fingers. Keep squeezing indefinitely
            # (the periodic timer above just keeps republishing this)
            # instead of relaxing once the action call returns, since the
            # arm will move the object around right after this returns.
            self._commanded_effort = self.hold_effort
            self.get_logger().info(
                f"Gripper stalled at {final_position:.3f} rad (target {target_position:.3f}) "
                f"-- object detected, holding at {self.hold_effort} Nm"
            )
        elif closing:
            # Reached fully closed without ever stalling: closed on nothing.
            self._commanded_effort = self.rest_open_effort
            self.get_logger().warn("Gripper closed fully without stalling -- likely grasped nothing")
        else:
            # Opening: keep a gentle opening torque so the fingers rest
            # against the open stop instead of sagging under gravity.
            self._commanded_effort = self.rest_open_effort

        result = GripperCommand.Result()
        result.position = final_position
        result.effort = final_effort
        result.stalled = stalled
        result.reached_goal = True
        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = GripperEffortController()
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
