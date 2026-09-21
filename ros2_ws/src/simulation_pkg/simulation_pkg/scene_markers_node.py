#!/usr/bin/env python3
"""ROS2 node that publishes RViz2 markers for the table, sugar box and
camera so they appear in RViz2 alongside the robot arm, since Gazebo's
physical simulation is not visible there directly."""

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3, Pose, Point, PoseStamped
from std_msgs.msg import ColorRGBA, String

# The grasp executor reports these while the box is held by the gripper.
ATTACHED_STATES = {'LIFTING', 'TRANSPORTING', 'PLACING'}
# After release the box stays where it was set down until perception looks again.
HOLD_STATES = {'RELEASING', 'RETREATING'}
BOX_HALF_HEIGHT = 0.088


class SceneMarkersNode(Node):
    """Publishes RViz2 markers for the table and camera, plus a sugar box
    marker that tracks the real box (see _publish_box)."""

    def __init__(self):
        super().__init__('scene_markers_node')

        self.publisher = self.create_publisher(
            MarkerArray, '/scene_markers', 10
        )

        # The box marker follows the real box: the perceived pose while it is
        # free, the gripper while it is attached (shown in a different
        # colour), and its resting spot after it has been released.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.status = 'IDLE'
        # start at the spawn pose from table_scene.sdf until perception reports one
        self.box_T = self._to_T([0.8, 0.0, 0.838], Rotation.from_euler('z', 0.3).as_quat())  # world <- box centre shown
        self.rel_T = None          # box pose relative to the gripper while attached
        self.create_subscription(PoseStamped, '/object_pose', self._pose_cb, 10)
        self.create_subscription(String, '/grasp_status', self._status_cb, 10)
        self.box_timer = self.create_timer(0.1, self._publish_box)

        # publish repeatedly so a late-starting RViz2 still receives these static markers
        self.timer = self.create_timer(1.0, self._publish_markers)
        self.get_logger().info("Scene markers node started")


    @staticmethod
    def _to_T(pos, quat_xyzw):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
        T[:3, 3] = pos
        return T

    def _tcp_T(self):
        try:
            t = self.tf_buffer.lookup_transform('world', 'tcp_link', rclpy.time.Time())
        except Exception:
            return None
        p, q = t.transform.translation, t.transform.rotation
        return self._to_T([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])

    def _pose_cb(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        # ignore poses that are not a box resting on the table (FoundationPose
        # can emit garbage while initialising, and the box is high while carried)
        if not (0.5 < p.x < 1.1 and -0.35 < p.y < 0.45 and 0.70 < p.z < 0.82):
            return
        if self.status in ATTACHED_STATES or self.status in HOLD_STATES:
            return
        # the perceived pose is the box's bottom face; show its centre
        T = self._to_T([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        T[:3, 3] += T[:3, 2] * BOX_HALF_HEIGHT
        self.box_T = T

    def _status_cb(self, msg: String):
        prev, self.status = self.status, msg.data
        if self.status in ATTACHED_STATES and prev not in ATTACHED_STATES:
            tcp = self._tcp_T()
            if tcp is not None and self.box_T is not None:
                self.rel_T = np.linalg.inv(tcp) @ self.box_T
        elif self.status not in ATTACHED_STATES:
            self.rel_T = None

    def _publish_box(self):
        if self.box_T is None:
            return
        attached = self.status in ATTACHED_STATES and self.rel_T is not None
        if attached:
            tcp = self._tcp_T()
            if tcp is not None:
                self.box_T = tcp @ self.rel_T
        q = Rotation.from_matrix(self.box_T[:3, :3]).as_quat()

        box = Marker()
        box.header.frame_id = 'world'
        box.header.stamp = self.get_clock().now().to_msg()
        box.ns = 'scene'
        box.id = 1
        box.type = Marker.CUBE
        box.action = Marker.ADD
        box.pose.position.x, box.pose.position.y, box.pose.position.z = [float(v) for v in self.box_T[:3, 3]]
        box.pose.orientation.x, box.pose.orientation.y, box.pose.orientation.z, box.pose.orientation.w = [float(v) for v in q]
        box.scale.x, box.scale.y, box.scale.z = 0.0495, 0.0942, 0.176
        if attached:
            box.color.r, box.color.g, box.color.b = 0.15, 0.75, 0.3   # attached: green
        else:
            box.color.r, box.color.g, box.color.b = 0.8, 0.2, 0.1     # free: orange
        box.color.a = 1.0
        self.publisher.publish(MarkerArray(markers=[box]))

    def _publish_markers(self):
        markers = MarkerArray()

        # table top
        table_top = Marker()
        table_top.header.frame_id = 'world'
        table_top.header.stamp = self.get_clock().now().to_msg()
        table_top.ns = 'scene'
        table_top.id = 0
        table_top.type = Marker.CUBE
        table_top.action = Marker.ADD
        table_top.pose.position.x = 0.8
        table_top.pose.position.y = 0.0
        table_top.pose.position.z = 0.725
        table_top.pose.orientation.w = 1.0
        table_top.scale.x = 1.2
        table_top.scale.y = 0.8
        table_top.scale.z = 0.05
        table_top.color.r = 0.9
        table_top.color.g = 0.9
        table_top.color.b = 0.85
        table_top.color.a = 0.9
        table_top.lifetime.sec = 0
        markers.markers.append(table_top)

        leg_positions = [
            (0.8 + 0.54, 0.34),
            (0.8 + 0.54, -0.34),
            (0.8 - 0.54, 0.34),
            (0.8 - 0.54, -0.34),
        ]
        for i, (lx, ly) in enumerate(leg_positions):
            leg = Marker()
            leg.header.frame_id = 'world'
            leg.header.stamp = self.get_clock().now().to_msg()
            leg.ns = 'scene'
            leg.id = 10 + i
            leg.type = Marker.CUBE
            leg.action = Marker.ADD
            leg.pose.position.x = lx
            leg.pose.position.y = ly
            leg.pose.position.z = 0.35
            leg.pose.orientation.w = 1.0
            leg.scale.x = 0.05
            leg.scale.y = 0.05
            leg.scale.z = 0.70
            leg.color.r = 0.7
            leg.color.g = 0.7
            leg.color.b = 0.65
            leg.color.a = 0.9
            leg.lifetime.sec = 0
            markers.markers.append(leg)

        # camera
        camera = Marker()
        camera.header.frame_id = 'world'
        camera.header.stamp = self.get_clock().now().to_msg()
        camera.ns = 'scene'
        camera.id = 2
        camera.type = Marker.CUBE
        camera.action = Marker.ADD

        camera.pose.position.x = 0.8
        camera.pose.position.y = 0.5
        camera.pose.position.z = 1.45
        camera.pose.orientation.w = 1.0

        # Small box representing camera body
        camera.scale.x = 0.025
        camera.scale.y = 0.09
        camera.scale.z = 0.025

        # Dark grey like a real RealSense
        camera.color.r = 0.2
        camera.color.g = 0.2
        camera.color.b = 0.2
        camera.color.a = 1.0

        camera.lifetime.sec = 0
        markers.markers.append(camera)

        self.publisher.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = SceneMarkersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()