#!/usr/bin/env python3
"""
SAM2 Segmentation Node
----------------------
Subscribes to the RGB camera topic, runs SAM2 to segment the target
object, and publishes a binary mask for FoundationPose to use during
pose initialization.

Topics subscribed:
  /camera/color/image_raw     (sensor_msgs/Image)

Topics published:
  /object_mask                (sensor_msgs/Image)  — binary mask
  /object_mask/debug          (sensor_msgs/Image)  — color visualization
  /sam2/status                (std_msgs/String)     — current state
"""

import sys
import os
import numpy as np
import cv2
import rclpy # type: ignore
from rclpy.node import Node # type: ignore
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy # type: ignore

from sensor_msgs.msg import Image # type: ignore
from std_msgs.msg import String # type: ignore
from cv_bridge import CvBridge # type: ignore

# Add SAM2 to Python path
SAM2_PATH = os.path.join(
    os.path.dirname(__file__), '../../../../..', 'sam2'
)
sys.path.insert(0, os.path.abspath(SAM2_PATH))

try:
    import torch
    from sam2.build_sam import build_sam2 # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor # type: ignore
    SAM2_AVAILABLE = True
    print("[SAM2] SAM2 imported successfully")
except ImportError as e:
    print(f"[SAM2 WARN] SAM2 not importable: {e}")
    print("[SAM2 WARN] Running in MOCK MODE — publishing full-image mask")
    SAM2_AVAILABLE = False


class SAM2Node(Node):
    """
    ROS2 node that wraps SAM2 for automatic object segmentation.

    On each color frame it runs SAM2 with a center-point prompt and
    publishes the resulting mask. The mask is consumed by the
    FoundationPose node for pose initialization.

    The node runs at a lower rate than 30Hz (configurable, default 5Hz)
    because SAM2 is slower than FoundationPose tracking. We only need
    a good mask for initialization — we don't need to re-segment every
    single frame.
    """

    def __init__(self):
        super().__init__('sam2_node')

        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('model_config', 'sam2.1_hiera_small')
        self.declare_parameter('segmentation_rate_hz', 5.0)
        
        # The point prompt — where in the image to look for the object.
        # Default is image center (0.5, 0.5) as normalized coordinates.
        self.declare_parameter('prompt_point_x', 0.5)
        self.declare_parameter('prompt_point_y', 0.5)
        self.declare_parameter('confidence_threshold', 0.8)

        checkpoint = self.get_parameter('checkpoint_path').value
        model_cfg = self.get_parameter('model_config').value
        rate_hz = self.get_parameter('segmentation_rate_hz').value
        self.prompt_x = self.get_parameter('prompt_point_x').value
        self.prompt_y = self.get_parameter('prompt_point_y').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value

        self.get_logger().info("SAM2 node starting...")
        self.get_logger().info(f"  Checkpoint: {checkpoint}")
        self.get_logger().info(
            f"  Prompt point: ({self.prompt_x}, {self.prompt_y})"
        )
        self.get_logger().info(f"  Rate: {rate_hz}Hz")

        self.bridge = CvBridge()
        self.predictor = None
        self.latest_color_image = None  # stores most recent frame
        self.frame_count = 0

        if SAM2_AVAILABLE and checkpoint and os.path.exists(checkpoint):
            self._load_model(checkpoint, model_cfg)
        else:
            self.get_logger().warn(
                f"Checkpoint not found at '{checkpoint}', using mock mode"
            )

        # BEST_EFFORT matches Gazebo's publisher QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # We subscribe to color images and store the latest one.
        # The actual segmentation runs on a timer to avoid blocking the executor.
        self.color_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self._color_callback,
            qos
        )

        # Binary mask — white where object is, black everywhere else
        self.mask_pub = self.create_publisher(Image, '/object_mask', 10)
        
        # Debug visualization — color image with mask overlay (for humans)
        self.debug_pub = self.create_publisher(Image, '/object_mask/debug', 10)
        
        # Status string for terminal feedback
        self.status_pub = self.create_publisher(String, '/sam2/status', 10)

        # Timer fires at 'rate_hz' (default 5Hz). Each time it fires, 
        # it segments whatever the latest image is.
        period = 1.0 / rate_hz
        self.timer = self.create_timer(period, self._segmentation_timer)

        self.get_logger().info(
            f"SAM2 node ready. Segmenting at {rate_hz}Hz"
        )

    def _load_model(self, checkpoint: str, model_cfg: str):
        """Load SAM2 model onto GPU."""
        try:
            self.get_logger().info("Loading SAM2 model onto GPU...")

            # Map the short model name to the full config path
            config_map = {
                'sam2.1_hiera_tiny':  'configs/sam2.1/sam2.1_hiera_t.yaml',
                'sam2.1_hiera_small': 'configs/sam2.1/sam2.1_hiera_s.yaml',
                'sam2.1_hiera_base':  'configs/sam2.1/sam2.1_hiera_b+.yaml',
                'sam2.1_hiera_large': 'configs/sam2.1/sam2.1_hiera_l.yaml',
            }

            config_path = config_map.get(model_cfg)
            if config_path is None:
                raise ValueError(f"Unknown model config: {model_cfg}")

            sam2_model = build_sam2(
                config_file=config_path,
                ckpt_path=checkpoint,
                device='cuda'
            )

            # SAM2ImagePredictor is the API for single-image segmentation
            self.predictor = SAM2ImagePredictor(sam2_model)

            self.get_logger().info("SAM2 model loaded successfully on GPU!")

        except Exception as e:
            self.get_logger().error(f"Failed to load SAM2: {e}")
            self.get_logger().warn("Falling back to mock mode")
            self.predictor = None

    def _color_callback(self, msg: Image):
        """
        Store the latest image without processing. 
        The timer will pick it up for segmentation.
        """
        self.latest_color_image = msg

    def _segmentation_timer(self):
        """Main segmentation callback — runs at segmentation_rate_hz (5Hz)."""
        if self.latest_color_image is None:
            return

        self.frame_count += 1

        # SAM2 expects RGB uint8 (H, W, 3)
        color_image = self.bridge.imgmsg_to_cv2(
            self.latest_color_image, desired_encoding='rgb8'
        )

        if self.predictor is not None:
            mask, confidence = self._run_sam2(color_image)
        else:
            mask, confidence = self._mock_mask(color_image)

        if mask is not None:
            self._publish_mask(mask, self.latest_color_image.header)
            self._publish_debug(color_image, mask, self.latest_color_image.header)

            status = String()
            status.data = (
                f"Frame {self.frame_count} | "
                f"Confidence: {confidence:.3f} | "
                f"Mask pixels: {np.sum(mask > 0)}"
            )
            self.status_pub.publish(status)

            if self.frame_count % 10 == 0:
                self.get_logger().info(status.data)

    def _run_sam2(self, image: np.ndarray):
        """Run SAM2 on a single image with a point prompt."""
        try:
            h, w = image.shape[:2]

            # Convert normalized prompt to pixel coordinates
            point_x = int(self.prompt_x * w)
            point_y = int(self.prompt_y * h)

            # point_labels: 1 = foreground, 0 = background
            point_coords = np.array([[point_x, point_y]])
            point_labels = np.array([1])

            # torch.no_grad() disables gradient tracking to save memory during inference
            with torch.no_grad():
                self.predictor.set_image(image) # type: ignore

                # multimask_output=True returns 3 candidate masks at different scales
                masks, scores, logits = self.predictor.predict( # type: ignore
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True
                )

            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])

            if best_score < self.conf_threshold:
                self.get_logger().warn(
                    f"SAM2 confidence {best_score:.3f} below "
                    f"threshold {self.conf_threshold}"
                )
                return None, best_score

            # FoundationPose expects uint8 mask where 255 = object
            mask_uint8 = best_mask.astype(np.uint8) * 255

            return mask_uint8, best_score

        except Exception as e:
            self.get_logger().error(f"SAM2 error: {e}")
            return None, 0.0

    def _mock_mask(self, image: np.ndarray):
        """
        Generate a mock mask when SAM2 is not available.
        Creates an ellipse in the center of the image.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        center = (int(self.prompt_x * w), int(self.prompt_y * h))
        axes = (w // 8, h // 6)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1) # type: ignore

        return mask, 1.0

    def _publish_mask(self, mask: np.ndarray, header):
        """
        Publish the binary mask as a ROS2 Image message.
        encoding='mono8' (0 = background, 255 = object).
        """
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = header
        self.mask_pub.publish(mask_msg)

    def _publish_debug(self, image: np.ndarray, mask: np.ndarray, header):
        """
        Publish a visualization showing the mask overlaid on the color image.
        Overlays the mask in green with 50% transparency.
        """
        debug_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        overlay = debug_img.copy()
        overlay[mask > 0] = [0, 255, 0]
        
        # Blend original and overlay
        debug_img = cv2.addWeighted(debug_img, 0.6, overlay, 0.4, 0)

        # Draw the prompt point as a red dot
        h, w = image.shape[:2]
        point_x = int(self.prompt_x * w)
        point_y = int(self.prompt_y * h)
        cv2.circle(debug_img, (point_x, point_y), 8, (0, 0, 255), -1)
        cv2.circle(debug_img, (point_x, point_y), 8, (255, 255, 255), 2)

        cv2.putText(
            debug_img, f"SAM2 Frame {self.frame_count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )

        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SAM2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()