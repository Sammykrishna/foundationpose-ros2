#!/usr/bin/env python3
"""ROS2 node that animates the UR5e through the grasp sequence by directly
publishing joint states, based on /grasp_status, without needing full
MoveIt2 infrastructure."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import numpy as np
import math


# UR5e joint names in order
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
]

# upright home position
HOME = [0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0]

# reaching forward and down toward the table
PRE_GRASP = [0.0, -1.2, 1.8, -2.15, -1.5708, 0.0]

# further down at object level
GRASPING = [0.0, -1.0, 2.0, -2.57, -1.5708, 0.0]

# lifted up with the object
LIFTING = [0.0, -1.2, 1.8, -2.15, -1.5708, 0.0]


def interpolate(q_start, q_end, t):
    """Interpolate between two joint configurations with an ease-in-out curve."""
    t_smooth = 3 * t**2 - 2 * t**3
    return [s + (e - s) * t_smooth for s, e in zip(q_start, q_end)]


class TrajectoryAnimator(Node):
    """Animates the UR5e through the grasp sequence based on /grasp_status."""

    def __init__(self):
        super().__init__('trajectory_animator')

        self.get_logger().info("Trajectory animator starting...")

        self.current_joints = HOME.copy()
        self.target_joints = HOME.copy()
        self.start_joints = HOME.copy()

        self.animation_progress = 1.0  # 1.0 = done, 0.0 = just started
        self.animation_duration = 2.0  # seconds per motion segment
        self.last_time = self.get_clock().now()
        self.current_status = 'IDLE'

        self.joint_pub = self.create_publisher(
            JointState, '/joint_states', 10
        )

        self.status_sub = self.create_subscription(
            String,
            '/grasp_status',
            self._status_callback,
            10
        )

        self.timer = self.create_timer(1.0/30.0, self._animation_step)

        self.get_logger().info(
            "Trajectory animator ready. Waiting for grasp status..."
        )

    def _status_callback(self, msg: String):
        """React to grasp state changes by setting a new animation target."""
        new_status = msg.data

        # only react to state changes, not repeated same state
        if new_status == self.current_status:
            return

        self.current_status = new_status
        self.get_logger().info(f"Grasp status: {new_status}, animating")

        state_to_config = {
            'MOVING_HOME': HOME,
            'PRE_GRASP':   PRE_GRASP,
            'GRASPING':    GRASPING,
            'LIFTING':     LIFTING,
            'DONE':        HOME,
            'IDLE':        None,    # don't move on IDLE
            'ERROR':       HOME,
        }

        target = state_to_config.get(new_status)

        if target is not None:
            self.start_joints = self.current_joints.copy()
            self.target_joints = target.copy()
            self.animation_progress = 0.0
            self.last_time = self.get_clock().now()

    def _animation_step(self):
        """Advance the animation and publish the current joint state, called at 30Hz."""
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if self.animation_progress < 1.0:
            self.animation_progress = min(
                1.0,
                self.animation_progress + dt / self.animation_duration
            )
            self.current_joints = interpolate(
                self.start_joints,
                self.target_joints,
                self.animation_progress
            )

        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = JOINT_NAMES
        msg.position = self.current_joints
        msg.velocity = [0.0] * 6
        msg.effort = [0.0] * 6

        self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryAnimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()