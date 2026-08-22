#!/usr/bin/env python3
"""
Planning Scene Manager Node
----------------------------
Keeps MoveIt2's planning scene in sync with the physical scene:
  - Table: added once as a static collision object (slab + 4 legs),
    matching scene_markers_node.py / table_scene.sdf exactly.
  - Sugar box: updated live from /object_pose, using the true
    geometric center rather than FoundationPose's bottom-center
    convention.

Topics subscribed:
  /object_pose       (geometry_msgs/PoseStamped)   from FoundationPose

Topics published:
  collision_object   (moveit_msgs/CollisionObject) to move_group
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String

from robot_control_pkg.box_geometry_utils import (
    get_box_center_from_bottom_pose,
    SUGAR_BOX_SIZE,
    SUGAR_BOX_HALF_HEIGHT,
)

# Must match scene_markers_node.py exactly
TABLE_TOP_POSE = (0.8, 0.0, 0.725)
TABLE_TOP_SIZE = (1.2, 0.8, 0.05)
LEG_SIZE = (0.05, 0.05, 0.70)
LEG_Z = 0.35
LEG_XY_OFFSETS = [
    (0.8 + 0.54, 0.34),
    (0.8 + 0.54, -0.34),
    (0.8 - 0.54, 0.34),
    (0.8 - 0.54, -0.34),
]

ATTACHED_STATES = {'CLOSING_GRIPPER', 'LIFTING', 'RETURNING_HOME'}


class PlanningSceneManagerNode(Node):
    def __init__(self):
        super().__init__('planning_scene_manager')

        qos = QoSProfile(depth=10)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.collision_pub = self.create_publisher(CollisionObject, 'collision_object', qos)
        self.pose_sub = self.create_subscription(PoseStamped, '/object_pose', self._pose_callback, 10)
        self.latest_object_pose = None

        self.box_is_attached = False
        self.status_sub = self.create_subscription(
            String, '/grasp_status', self._grasp_status_callback, 10
        )

        # Table is static — resend periodically so a late-starting
        # move_group still picks it up
        self.table_timer = self.create_timer(2.0, self._publish_table)
        # Box — throttled, not every incoming pose message
        self.box_timer = self.create_timer(0.2, self._publish_box)  # 5 Hz

        self.get_logger().info("Planning scene manager started")

    def _pose_callback(self, msg: PoseStamped):
        self.latest_object_pose = msg

    def _grasp_status_callback(self, msg: String):
        self.box_is_attached = msg.data in ATTACHED_STATES

    def _publish_table(self):
        table = CollisionObject()
        table.header.frame_id = 'world'
        table.header.stamp = self.get_clock().now().to_msg()
        table.id = 'table'
        table.operation = CollisionObject.ADD

        slab = SolidPrimitive()
        slab.type = SolidPrimitive.BOX
        slab.dimensions = list(TABLE_TOP_SIZE)
        slab_pose = Pose()
        slab_pose.position.x, slab_pose.position.y, slab_pose.position.z = TABLE_TOP_POSE
        slab_pose.orientation.w = 1.0
        table.primitives.append(slab)
        table.primitive_poses.append(slab_pose)

        for lx, ly in LEG_XY_OFFSETS:
            leg = SolidPrimitive()
            leg.type = SolidPrimitive.BOX
            leg.dimensions = list(LEG_SIZE)
            leg_pose = Pose()
            leg_pose.position.x, leg_pose.position.y, leg_pose.position.z = lx, ly, LEG_Z
            leg_pose.orientation.w = 1.0
            table.primitives.append(leg)
            table.primitive_poses.append(leg_pose)

        self.collision_pub.publish(table)

    def _publish_box(self):
        if self.latest_object_pose is None or self.box_is_attached:
            return

        center_pose = get_box_center_from_bottom_pose(
            self.latest_object_pose.pose, SUGAR_BOX_HALF_HEIGHT
        )

        box = CollisionObject()
        box.header.frame_id = self.latest_object_pose.header.frame_id or 'world'
        box.header.stamp = self.get_clock().now().to_msg()
        box.id = 'sugar_box'
        box.operation = CollisionObject.ADD

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(SUGAR_BOX_SIZE)
        box.primitives.append(primitive)
        box.primitive_poses.append(center_pose)

        self.collision_pub.publish(box)


def main(args=None):
    rclpy.init(args=args)
    node = PlanningSceneManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()