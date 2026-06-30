#!/usr/bin/env python3
"""
Trajectory Animator Node
------------------------
Animates the UR5e robot through the grasp sequence by directly
publishing joint states. This gives the visual effect of the robot
moving through HOME → PRE_GRASP → GRASPING → LIFTING without
needing full MoveIt2 infrastructure.

Subscribes to /grasp_status to know when to animate.
Publishes to /joint_states to move the robot in RViz2.
"""

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

# Key robot configurations as joint angles (radians)
# These are real UR5e configurations that look natural

# Upright home position
HOME = [0.0, -1.5708, 1.5708, -1.5708, 0.0, 0.0]

# Reaching forward and down toward table (pre-grasp)
# shoulder_pan rotated to face table, arm extended forward
PRE_GRASP = [0.0, -1.2, 1.8, -2.15, -1.5708, 0.0]

# Further down at object level (grasp)
GRASPING = [0.0, -1.0, 2.0, -2.57, -1.5708, 0.0]

# Lifted up with object (post-grasp)
LIFTING = [0.0, -1.2, 1.8, -2.15, -1.5708, 0.0]


def interpolate(q_start, q_end, t):
    """
    Linear interpolation between two joint configurations.
    t is between 0.0 and 1.0.
    We use a smooth ease-in-out curve so the motion looks natural.
    """
    # Smooth step: 3t^2 - 2t^3 gives ease-in-out
    t_smooth = 3 * t**2 - 2 * t**3
    return [s + (e - s) * t_smooth for s, e in zip(q_start, q_end)]


class TrajectoryAnimator(Node):
    """
    Animates UR5e through grasp sequence based on /grasp_status topic.

    When grasp_status changes:
    - MOVING_HOME  → animate to HOME
    - PRE_GRASP    → animate to PRE_GRASP
    - GRASPING     → animate to GRASPING
    - LIFTING      → animate to LIFTING (then back to HOME)
    - IDLE/DONE    → hold current position
    """

    def __init__(self):
        super().__init__('trajectory_animator')

        self.get_logger().info("Trajectory animator starting...")

        # Current joint state (start at home)
        self.current_joints = HOME.copy()
        self.target_joints = HOME.copy()
        self.start_joints = HOME.copy()

        # Animation state
        self.animation_progress = 1.0  # 1.0 = done, 0.0 = just started
        self.animation_duration = 2.0  # seconds per motion segment
        self.last_time = self.get_clock().now()
        self.current_status = 'IDLE'

        # Publisher — sends joint states to robot_state_publisher
        # which then updates TF and makes robot move in RViz2
        self.joint_pub = self.create_publisher(
            JointState, '/joint_states', 10
        )

        # Subscriber — listens to grasp executor state machine
        self.status_sub = self.create_subscription(
            String,
            '/grasp_status',
            self._status_callback,
            10
        )

        # Animation loop at 30Hz — smooth motion
        self.timer = self.create_timer(1.0/30.0, self._animation_step)

        self.get_logger().info(
            "Trajectory animator ready. Waiting for grasp status..."
        )

    def _status_callback(self, msg: String):
        """
        React to grasp state changes by setting new animation targets.
        Each state has a corresponding robot configuration.
        """
        new_status = msg.data

        # Only react to state changes, not repeated same state
        if new_status == self.current_status:
            return

        self.current_status = new_status
        self.get_logger().info(f"Grasp status: {new_status} → animating")

        # Map each state to its target configuration
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
            # Start a new animation from current position to target
            self.start_joints = self.current_joints.copy()
            self.target_joints = target.copy()
            self.animation_progress = 0.0
            self.last_time = self.get_clock().now()

    def _animation_step(self):
        """
        Called at 30Hz. Advances animation and publishes joint states.
        """
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # Advance animation progress
        if self.animation_progress < 1.0:
            self.animation_progress = min(
                1.0,
                self.animation_progress + dt / self.animation_duration
            )
            # Interpolate between start and target
            self.current_joints = interpolate(
                self.start_joints,
                self.target_joints,
                self.animation_progress
            )

        # Publish current joint state
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