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

# ROS2 message types we use
from sensor_msgs.msg import Image, CameraInfo # type: ignore
from geometry_msgs.msg import PoseStamped # type: ignore
from visualization_msgs.msg import Marker # type: ignore

# cv_bridge converts between ROS2 Image messages and OpenCV numpy arrays
# Without this we'd have to manually parse the raw byte buffer
from cv_bridge import CvBridge # type: ignore

# message_filters lets us synchronize two topics by timestamp
# We need color and depth images taken at the SAME moment
# If we subscribed to them independently, we might pair a color frame
# from t=1.000s with a depth frame from t=1.033s — that mismatch
# would give FoundationPose bad data and wrong poses
import message_filters # type: ignore

# scipy for converting rotation matrix -> quaternion
# ROS2 uses quaternions for orientation, FoundationPose gives a 4x4 matrix
from scipy.spatial.transform import Rotation

# FoundationPose imports — these come from the cloned repo
# We add it to the path so Python can find it
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

        # ---- Parameters ----
        # Using ROS2 parameters means you can change these from the
        # command line or launch file without editing code
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

        # ---- Internal state ----
        self.bridge = CvBridge()
        self.camera_intrinsics = None   # filled when first CameraInfo arrives
        self.estimator = None           # FoundationPose model instance
        self.mesh = None                # trimesh object
        self.is_initialized = False     # False = need to run initialization
        self.frame_count = 0

        # ---- Load mesh ----
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

        # ---- Load FoundationPose model ----
        if FOUNDATIONPOSE_AVAILABLE and self.mesh is not None and weights_dir:
            self._load_model(weights_dir)

        # ---- QoS Profile ----
        # BEST_EFFORT matches Gazebo's publisher QoS
        # If we used RELIABLE here and Gazebo uses BEST_EFFORT,
        # ROS2 would refuse to connect them — a common gotcha
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # we only care about the latest frame, not a backlog
        )

        # ---- Subscribers ----
        # We subscribe to CameraInfo separately (not synchronized)
        # because it barely changes — we just need it once to get K matrix
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self._camera_info_callback,
            qos
        )

        # Synchronized subscribers for color + depth
        # message_filters.Subscriber wraps a normal ROS2 subscription
        # so it can be fed into the synchronizer
        color_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw',
            qos_profile=qos
        )
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw',
            qos_profile=qos
        )

        # ApproximateTimeSynchronizer pairs messages whose timestamps
        # are within 'slop' seconds of each other
        # queue_size=5 means it buffers up to 5 messages while waiting for a match
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=5,
            slop=0.05  # 50ms tolerance — fine for 30Hz camera
        )
        self.sync.registerCallback(self._rgbd_callback)

        # ---- Publishers ----
        self.pose_pub = self.create_publisher(
            PoseStamped, '/object_pose', 10
        )
        self.marker_pub = self.create_publisher(
            Marker, '/pose_marker', 10
        )

        self.get_logger().info("FoundationPose node ready. Waiting for camera data...")

    def _load_model(self, weights_dir):
        """Load the FoundationPose neural network weights."""
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
        """
        Extract the camera intrinsic matrix K from CameraInfo.
        
        K is a 3x3 matrix:
        [fx  0  cx]
        [ 0 fy  cy]
        [ 0  0   1]
        
        fx, fy = focal lengths in pixels
        cx, cy = principal point (optical center) in pixels
        
        FoundationPose needs K to convert the 2D depth image into
        a 3D point cloud — it uses K to "unproject" each pixel.
        """
        if self.camera_intrinsics is None:
            K = np.array(msg.k).reshape(3, 3)
            self.camera_intrinsics = K
            self.get_logger().info(
                f"Camera intrinsics received:\n"
                f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}\n"
                f"  cx={K[0,2]:.1f}, cy={K[1,2]:.1f}"
            )

    def _rgbd_callback(self, color_msg: Image, depth_msg: Image):
        """
        Main callback — called every time we get a synchronized color+depth pair.
        This runs at 30Hz.
        """
        # Don't process until we have camera intrinsics
        if self.camera_intrinsics is None:
            return

        self.frame_count += 1

        # ---- Convert ROS2 messages to numpy arrays ----
        # cv_bridge handles the encoding conversion for us
        # color: ROS BGR8 -> numpy (H, W, 3) uint8
        color_image = self.bridge.imgmsg_to_cv2(
            color_msg, desired_encoding='bgr8'
        )
        # depth: ROS 32FC1 (float meters) -> numpy (H, W) float32
        depth_image = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='32FC1'
        )

        # ---- Estimate pose ----
        if self.estimator is not None:
            pose_matrix = self._run_foundationpose(color_image, depth_image)
        else:
            # Mock mode: return a static pose above the table
            # This is useful for testing the ROS2 pipeline end-to-end
            # before the model is fully set up
            pose_matrix = self._mock_pose()

        # ---- Publish results ----
        if pose_matrix is not None:
            self._publish_pose(pose_matrix, color_msg.header)

        # Log progress every 30 frames (once per second at 30Hz)
        if self.frame_count % 30 == 0:
            mode = "TRACKING" if self.is_initialized else "INITIALIZING"
            self.get_logger().info(
                f"Frame {self.frame_count} | Mode: {mode} | "
                f"Pose valid: {pose_matrix is not None}"
            )

    def _run_foundationpose(self, color: np.ndarray, depth: np.ndarray):
        """
        Run FoundationPose on one RGB-D frame.
        Returns a 4x4 numpy pose matrix, or None if estimation failed.
        """
        try:
            # FoundationPose expects RGB not BGR (OpenCV default is BGR)
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            # Depth needs to be in meters as float32
            # Gazebo publishes in meters already, so no conversion needed
            # A real RealSense publishes in millimeters, so you'd divide by 1000

            if not self.is_initialized:
                # INITIALIZATION MODE
                # We need an initial mask to tell FoundationPose roughly
                # where the object is. Later SAM2 will provide this automatically.
                # For now we generate a simple mask based on depth threshold.
                mask = self._generate_depth_mask(depth)

                poses = self.estimator.register( # pyright: ignore[reportOptionalMemberAccess]
                    K=self.camera_intrinsics,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=mask,
                    iteration=5   # more iterations = more accurate but slower
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
                # TRACKING MODE — much faster
                pose = self.estimator.track_one( # type: ignore
                    rgb=rgb,
                    depth=depth,
                    K=self.camera_intrinsics,
                    iteration=2   # fewer iterations needed for tracking
                )
                self.current_pose = pose
                return pose

        except Exception as e:
            self.get_logger().error(f"FoundationPose error: {e}")
            return None

    def _generate_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for initialization.
        
        We look for pixels where the depth value is consistent with
        an object sitting on the table (~0.75m from camera origin,
        adjusted for camera height).
        
        This is a simple heuristic. SAM2 (Step 4) will replace this
        with a proper learned segmentation.
        """
        # Camera is at z=1.45m, table top at z=0.75m
        # So depth to table surface ≈ 0.70m
        # Object is ~0.088m tall, so object depth ≈ 0.61m to 0.70m
        mask = np.logical_and(depth > 0.5, depth < 0.72).astype(np.uint8) * 255

        # Clean up noise with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove small dots
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes

        return mask

    def _mock_pose(self) -> np.ndarray:
        """
        Return a fake 4x4 pose matrix for testing the pipeline
        when FoundationPose is not yet loaded.
        
        Places the object at a fixed position in front of the camera.
        """
        pose = np.eye(4)  # identity matrix = no rotation, at origin
        pose[0, 3] = 0.8  # x = 0.8m (in front of camera)
        pose[1, 3] = 0.0  # y = 0.0m (centered)
        pose[2, 3] = 0.75 # z = 0.75m (table height)
        return pose

    def _publish_pose(self, pose_matrix: np.ndarray, header):
        """
        Convert a 4x4 pose matrix to ROS2 PoseStamped and publish it.
        
        A 4x4 transformation matrix looks like:
        [R R R tx]
        [R R R ty]   R = 3x3 rotation matrix
        [R R R tz]   t = translation vector (x, y, z)
        [0 0 0  1]
        
        ROS2 PoseStamped uses:
        - position: (x, y, z) in meters
        - orientation: quaternion (x, y, z, w)
        """
        # Extract translation from last column of matrix
        translation = pose_matrix[:3, 3]

        # Extract rotation matrix (top-left 3x3) and convert to quaternion
        rotation_matrix = pose_matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # type: ignore # returns [x, y, z, w]

        # Build PoseStamped message
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

        # Also publish a visual marker for RViz2
        # Without this you'd have no visual feedback that pose estimation is working
        self._publish_marker(pose_msg)

    def _publish_marker(self, pose_msg: PoseStamped):
        """
        Publish a green arrow marker in RViz2 showing the estimated pose.
        The arrow points in the object's +Z direction.
        """
        marker = Marker()
        marker.header = pose_msg.header
        marker.ns = 'object_pose'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose = pose_msg.pose

        # Arrow dimensions
        marker.scale.x = 0.1   # shaft length
        marker.scale.y = 0.01  # shaft diameter
        marker.scale.z = 0.01  # head diameter

        # Green color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # alpha=1 means fully opaque

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