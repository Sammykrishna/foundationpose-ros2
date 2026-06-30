#!/usr/bin/env python3
"""
Scene Markers Node
------------------
Publishes RViz2 markers for the table and sugar box so they
appear in RViz2 alongside the robot arm.

This bridges the gap between what exists in Gazebo (physical
simulation) and what RViz2 shows (sensor/planning data).

Topics published:
  /scene_markers  (visualization_msgs/MarkerArray)
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3, Pose, Point
from std_msgs.msg import ColorRGBA


class SceneMarkersNode(Node):
    """
    Publishes static markers for the table and sugar box.

    These match exactly the positions we defined in table_scene.sdf
    in Step 2, so RViz2 and Gazebo show the same scene layout.
    """

    def __init__(self):
        super().__init__('scene_markers_node')

        self.publisher = self.create_publisher(
            MarkerArray, '/scene_markers', 10
        )

        # Publish at 1Hz — these are static objects, no need for faster
        # We still publish repeatedly so RViz2 can receive them if it
        # starts after this node
        self.timer = self.create_timer(1.0, self._publish_markers)
        self.get_logger().info("Scene markers node started")

    def _publish_markers(self):
        markers = MarkerArray()

        # ---- TABLE ----
        # From our SDF: pose 0.8 0 0.375, size 1.2 x 0.8 x 0.75
        # The SDF pose is the CENTER of the box
        # so the table top surface is at z = 0.375 + 0.375 = 0.75m
        table = Marker()
        table.header.frame_id = 'world'
        table.header.stamp = self.get_clock().now().to_msg()
        table.ns = 'scene'
        table.id = 0
        table.type = Marker.CUBE
        table.action = Marker.ADD

        # Position — center of table box
        table.pose.position.x = 0.8
        table.pose.position.y = 0.0
        table.pose.position.z = 0.375
        table.pose.orientation.w = 1.0  # no rotation

        # Size — matches SDF exactly
        table.scale.x = 1.2   # length
        table.scale.y = 0.8   # width
        table.scale.z = 0.75  # height

        # Color — white/cream like our SDF table
        table.color.r = 0.9
        table.color.g = 0.9
        table.color.b = 0.85
        table.color.a = 0.9   # slightly transparent so we can see through

        # Lifetime 0 means marker persists until deleted
        table.lifetime.sec = 0

        markers.markers.append(table)

        # ---- SUGAR BOX ----
        # From our SDF: pose 0.8 0.0 0.794, rotation 0.3 rad around Z
        # Size: 0.038 x 0.057 x 0.088 (real YCB sugar box dimensions)
        import math
        sugar_box = Marker()
        sugar_box.header.frame_id = 'world'
        sugar_box.header.stamp = self.get_clock().now().to_msg()
        sugar_box.ns = 'scene'
        sugar_box.id = 1
        sugar_box.type = Marker.CUBE
        sugar_box.action = Marker.ADD

        # Position — center of sugar box, sitting on table top
        sugar_box.pose.position.x = 0.8
        sugar_box.pose.position.y = 0.0
        sugar_box.pose.position.z = 0.794

        # Rotation — 0.3 radians around Z axis (as set in SDF)
        # Convert to quaternion: q = [0, 0, sin(θ/2), cos(θ/2)]
        angle = 0.3
        sugar_box.pose.orientation.x = 0.0
        sugar_box.pose.orientation.y = 0.0
        sugar_box.pose.orientation.z = math.sin(angle / 2)
        sugar_box.pose.orientation.w = math.cos(angle / 2)

        # Real YCB sugar box dimensions in meters
        sugar_box.scale.x = 0.038
        sugar_box.scale.y = 0.057
        sugar_box.scale.z = 0.088

        # Color — red like our SDF sugar box
        sugar_box.color.r = 0.8
        sugar_box.color.g = 0.2
        sugar_box.color.b = 0.1
        sugar_box.color.a = 1.0

        sugar_box.lifetime.sec = 0
        markers.markers.append(sugar_box)

        # ---- CAMERA ----
        # Show where the RealSense camera is in the scene
        # From SDF: pose 0.8 -0.5 1.45
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