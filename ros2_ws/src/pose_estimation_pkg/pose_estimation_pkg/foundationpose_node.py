#!/usr/bin/env python3
"""
FoundationPose ROS2 Node
------------------------
Subscribes to RGB-D camera topics, runs FoundationPose to estimate
the 6-DoF pose of a target object, and publishes the result as a
geometry_msgs/PoseStamped on /object_pose.

Topics subscribed:
  /camera/color/image_raw      (sensor_msgs/Image)
  /camera/depth/image_raw      (sensor_msgs/Image)
  /camera/color/camera_info    (sensor_msgs/CameraInfo)
  /object_mask                 (sensor_msgs/Image)  <- from SAM2 node

Topics published:
  /object_pose                 (geometry_msgs/PoseStamped)
  /pose_marker                 (visualization_msgs/Marker)  <- for RViz2
"""

import sys
import os
import numpy as np
import cv2
import trimesh
import rclpy # type: ignore
from rclpy.node import Node # type: ignore
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy # type: ignore

from sensor_msgs.msg import Image, CameraInfo # type: ignore
from geometry_msgs.msg import PoseStamped # type: ignore
from visualization_msgs.msg import Marker # type: ignore
from cv_bridge import CvBridge # type: ignore
import message_filters # type: ignore
from scipy.spatial.transform import Rotation

FOUNDATIONPOSE_PATH = os.path.join(
    os.path.dirname(__file__), '../../../../..', 'FoundationPose'
)
sys.path.insert(0, os.path.abspath(FOUNDATIONPOSE_PATH))

try:
    from estimater import FoundationPose as FPEstimator # type: ignore
    from datareader import set_logging_format # type: ignore
    FOUNDATIONPOSE_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] FoundationPose not importable: {e}")
    print("[WARN] Running in MOCK MODE — publishing dummy poses for testing")
    FOUNDATIONPOSE_AVAILABLE = False


class FoundationPoseNode(Node):
    """
    ROS2 node that wraps FoundationPose for real-time 6-DoF pose estimation.
    
    On the first frame it runs pose INITIALIZATION (slow, ~1-2s).
    On every subsequent frame it runs pose TRACKING (fast, ~30ms).
    
    If FoundationPose is not available (import failed), it publishes
    a mock pose so you can test the rest of the pipeline.
    """

    def __init__(self):
        super().__init__('foundationpose_node')

        self.declare_parameter('mesh_path', '')
        self.declare_parameter('weights_dir', '')
        self.declare_parameter('score_threshold', 0.3)
        self.declare_parameter('debug_visualization', True)

        mesh_path = self.get_parameter('mesh_path').value
        weights_dir = self.get_parameter('weights_dir').value
        self.score_threshold = self.get_parameter('score_threshold').value
        self.debug_vis = self.get_parameter('debug_visualization').value

        self.get_logger().info("FoundationPose node starting...")
        self.get_logger().info(f"  Mesh: {mesh_path}")
        self.get_logger().info(f"  Weights: {weights_dir}")

        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self.estimator = None
        self.mesh = None
        self.is_initialized = False
        self.frame_count = 0

        if mesh_path and os.path.exists(mesh_path):
            self.get_logger().info("Loading mesh...")
            self.mesh = trimesh.load(mesh_path)
            self.get_logger().info(
                f"Mesh loaded: {len(self.mesh.vertices)} vertices" # type: ignore
            )
        else:
            self.get_logger().warn(
                f"Mesh not found at '{mesh_path}', will use mock pose"
            )

        if FOUNDATIONPOSE_AVAILABLE and self.mesh is not None and weights_dir:
            self._load_model(weights_dir)

        # BEST_EFFORT matches Gazebo's publisher QoS to avoid connection refusals.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self._camera_info_callback,
            qos
        )

        color_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw',
            qos_profile=qos
        )
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw',
            qos_profile=qos
        )

        # 50ms tolerance is fine for a 30Hz camera stream.
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=5,
            slop=0.05
        )
        self.sync.registerCallback(self._rgbd_callback)

        self.sam2_mask = None
        self.mask_sub = self.create_subscription(
            Image,
            '/object_mask',
            self._mask_callback,
            qos
        )
        self.get_logger().info("Subscribed to /object_mask from SAM2 node")

        self.pose_pub = self.create_publisher(
            PoseStamped, '/object_pose', 10
        )
        self.marker_pub = self.create_publisher(
            Marker, '/pose_marker', 10
        )

        self.get_logger().info("FoundationPose node ready. Waiting for camera data...")

    def _load_model(self, weights_dir):
        import torch
        from estimater import ScorePredictor, PoseRefinePredictor # type: ignore

        self.get_logger().info("Loading FoundationPose model weights...")

        try:
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()

            scorer_path = os.path.join(weights_dir, 'scorer', 'model_best.pth')
            refiner_path = os.path.join(weights_dir, 'refiner', 'model_best.pth')

            scorer.model.load_state_dict(
                torch.load(scorer_path, map_location='cuda')
            )
            refiner.model.load_state_dict(
                torch.load(refiner_path, map_location='cuda')
            )

            self.estimator = FPEstimator(
                scorer=scorer,
                refiner=refiner,
                debug=0,
                debug_dir='/tmp/foundationpose_debug'
            )
            self.get_logger().info("Model loaded successfully on GPU!")

        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.get_logger().warn("Falling back to mock mode")

    def _camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            K = np.array(msg.k).reshape(3, 3)
            self.camera_intrinsics = K
            self.get_logger().info(
                f"Camera intrinsics received:\n"
                f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}\n"
                f"  cx={K[0,2]:.1f}, cy={K[1,2]:.1f}"
            )

    def _mask_callback(self, msg: Image):
        """Cache the latest segmentation mask from the SAM2 node."""
        try:
            self.sam2_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f"Failed to decode SAM2 mask: {e}")

    def _rgbd_callback(self, color_msg: Image, depth_msg: Image):
        if self.camera_intrinsics is None:
            return

        self.frame_count += 1

        color_image = self.bridge.imgmsg_to_cv2(
            color_msg, desired_encoding='bgr8'
        )
        depth_image = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='32FC1'
        )

        if self.estimator is not None:
            pose_matrix = self._run_foundationpose(color_image, depth_image)
        else:
            pose_matrix = self._mock_pose()

        if pose_matrix is not None:
            self._publish_pose(pose_matrix, color_msg.header)

        if self.frame_count % 30 == 0:
            mode = "TRACKING" if self.is_initialized else "INITIALIZING"
            self.get_logger().info(
                f"Frame {self.frame_count} | Mode: {mode} | "
                f"Pose valid: {pose_matrix is not None}"
            )

    def _run_foundationpose(self, color: np.ndarray, depth: np.ndarray):
        try:
            # FoundationPose expects RGB, but OpenCV defaults to BGR.
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            # Note: Gazebo publishes depth in meters. A real RealSense 
            # publishes in millimeters, so you would need to divide by 1000.

            if not self.is_initialized:
                if self.sam2_mask is not None:
                    mask = self.sam2_mask
                else:
                    self.get_logger().warn(
                        "No SAM2 mask yet — falling back to depth threshold mask"
                    )
                    mask = self._generate_depth_mask(depth)

                poses = self.estimator.register( # pyright: ignore[reportOptionalMemberAccess]
                    K=self.camera_intrinsics,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=mask,
                    iteration=5
                )

                if poses is not None and len(poses) > 0:
                    self.is_initialized = True
                    self.current_pose = poses[0]
                    self.get_logger().info("Pose INITIALIZED successfully!")
                    return self.current_pose
                else:
                    self.get_logger().warn("Initialization failed, retrying...")
                    return None
            else:
                pose = self.estimator.track_one( # type: ignore
                    rgb=rgb,
                    depth=depth,
                    K=self.camera_intrinsics,
                    iteration=2
                )
                self.current_pose = pose
                return pose

        except Exception as e:
            self.get_logger().error(f"FoundationPose error: {e}")
            return None

    def _generate_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for initialization based on depth heuristics.
        Assumes camera is at z=1.45m, table top at z=0.75m, and object is ~0.088m tall.
        """
        mask = np.logical_and(depth > 0.5, depth < 0.72).astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _mock_pose(self) -> np.ndarray:
        """Return a fake 4x4 pose matrix for pipeline testing."""
        pose = np.eye(4)
        pose[0, 3] = 0.8  
        pose[1, 3] = 0.0  
        pose[2, 3] = 0.75 
        return pose

    def _publish_pose(self, pose_matrix: np.ndarray, header):
        """Convert a 4x4 pose matrix to a ROS2 PoseStamped and publish it."""
        translation = pose_matrix[:3, 3]
        rotation_matrix = pose_matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # type: ignore

        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = 'world'

        pose_msg.pose.position.x = float(translation[0])
        pose_msg.pose.position.y = float(translation[1])
        pose_msg.pose.position.z = float(translation[2])

        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])

        self.pose_pub.publish(pose_msg)
        self._publish_marker(pose_msg)

    def _publish_marker(self, pose_msg: PoseStamped):
        """Publish a green arrow marker in RViz2 showing the estimated pose."""
        marker = Marker()
        marker.header = pose_msg.header
        marker.ns = 'object_pose'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose_msg.pose

        marker.scale.x = 0.1   
        marker.scale.y = 0.01  
        marker.scale.z = 0.01  

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0  

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = FoundationPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()