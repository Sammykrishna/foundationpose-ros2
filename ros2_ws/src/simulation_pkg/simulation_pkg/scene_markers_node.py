#!/usr/bin/env python3
"""ROS2 node that publishes RViz2 markers for the table, sugar box and
camera so they appear in RViz2 alongside the robot arm, since Gazebo's
physical simulation is not visible there directly."""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3, Pose, Point
from std_msgs.msg import ColorRGBA


class SceneMarkersNode(Node):
    """Publishes static markers matching the positions in table_scene.sdf."""

    def __init__(self):
        super().__init__('scene_markers_node')

        self.publisher = self.create_publisher(
            MarkerArray, '/scene_markers', 10
        )

        # publish repeatedly so a late-starting RViz2 still receives these static markers
        self.timer = self.create_timer(1.0, self._publish_markers)
        self.get_logger().info("Scene markers node started")

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

        # sugar box
        import math
        sugar_box = Marker()
        sugar_box.header.frame_id = 'world'
        sugar_box.header.stamp = self.get_clock().now().to_msg()
        sugar_box.ns = 'scene'
        sugar_box.id = 1
        sugar_box.type = Marker.CUBE
        sugar_box.action = Marker.ADD

        sugar_box.pose.position.x = 0.8
        sugar_box.pose.position.y = 0.0
        sugar_box.pose.position.z = 0.838

        # convert the 0.3 rad Z rotation to a quaternion: q = [0, 0, sin(a/2), cos(a/2)]
        angle = 0.3
        sugar_box.pose.orientation.x = 0.0
        sugar_box.pose.orientation.y = 0.0
        sugar_box.pose.orientation.z = math.sin(angle / 2)
        sugar_box.pose.orientation.w = math.cos(angle / 2)

        # Real YCB sugar box dimensions in meters
        sugar_box.scale.x = 0.0495
        sugar_box.scale.y = 0.0942
        sugar_box.scale.z = 0.176

        sugar_box.color.r = 0.8
        sugar_box.color.g = 0.2
        sugar_box.color.b = 0.1
        sugar_box.color.a = 1.0

        sugar_box.lifetime.sec = 0
        markers.markers.append(sugar_box)

        # camera
        camera = Marker()
        camera.header.frame_id = 'world'
        camera.header.stamp = self.get_clock().now().to_msg()
        camera.ns = 'scene'
        camera.id = 2
        camera.type = Marker.CUBE
        camera.action = Marker.ADD

        camera.pose.position.x = 0.8
        camera.pose.position.y = -0.5
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