#!/usr/bin/env python3
"""ROS2 node that computes pose estimation accuracy (ADD, ATE, ARE) by
comparing FoundationPose output against Gazebo ground truth poses, and
saves the results to a JSON file."""

import os
import json
import time
import numpy as np
import trimesh
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from scipy.spatial.transform import Rotation


class BenchmarkNode(Node):
    """
    Computes and logs pose estimation accuracy metrics.

    Runs alongside the main pipeline, comparing estimated poses
    from FoundationPose against ground truth from Gazebo.
    """

    def __init__(self):
        super().__init__('benchmark_node')

        self.declare_parameter('mesh_path', '')
        self.declare_parameter('results_dir', '/tmp/benchmark_results')
        self.declare_parameter('num_samples', 200)
        self.declare_parameter('warmup_frames', 30)

        mesh_path = self.get_parameter('mesh_path').value
        self.results_dir = self.get_parameter('results_dir').value
        self.num_samples = self.get_parameter('num_samples').value
        self.warmup_frames = self.get_parameter('warmup_frames').value

        os.makedirs(self.results_dir, exist_ok=True)

        self.get_logger().info("Benchmark node starting...")
        self.get_logger().info(f"  Will collect {self.num_samples} samples")
        self.get_logger().info(f"  Results: {self.results_dir}")

        # Subsample 1000 points from the mesh for fast ADD computation
        self.mesh_points = None
        if mesh_path and os.path.exists(mesh_path):
            self.get_logger().info("Loading mesh for ADD computation...")
            mesh = trimesh.load(mesh_path)
            self.mesh_points, _ = trimesh.sample.sample_surface(mesh, 1000)
            self.get_logger().info(
                f"Mesh loaded: {len(self.mesh_points)} sample points"
            )

        self.estimated_poses = []
        self.frame_count = 0
        self.benchmark_active = False
        self.results_saved = False

        # Ground truth pose based on the SDF world file placement
        self.ground_truth_pose = self._build_ground_truth_pose()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/object_pose',
            self._pose_callback,
            qos
        )

        self.metrics_pub = self.create_publisher(
            String, '/benchmark/metrics', 10
        )

        # Print live metrics every 5 seconds
        self.metrics_timer = self.create_timer(5.0, self._print_metrics)

        self.get_logger().info(
            f"Benchmark ready. Warming up for {self.warmup_frames} frames..."
        )

    def _build_ground_truth_pose(self) -> np.ndarray:
        """Build the sugar box ground truth pose from its placement in the SDF world file."""
        r = Rotation.from_euler('z', 0.3)
        R = r.as_matrix()

        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = 0.8
        T[1, 3] = 0.0
        T[2, 3] = 0.794

        return T

    def _pose_to_matrix(self, pose_stamped: PoseStamped) -> np.ndarray:
        """Convert a ROS2 PoseStamped to a 4x4 numpy transformation matrix."""
        p = pose_stamped.pose.position
        q = pose_stamped.pose.orientation

        r = Rotation.from_quat([q.x, q.y, q.z, q.w])
        R = r.as_matrix()

        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = p.x
        T[1, 3] = p.y
        T[2, 3] = p.z

        return T

    def _compute_add(
        self,
        T_gt: np.ndarray,
        T_est: np.ndarray
    ) -> float:
        """Compute ADD, the average distance of transformed model points from ground truth."""
        if self.mesh_points is None:
            # fallback to translation error if no mesh is available
            return np.linalg.norm(T_gt[:3, 3] - T_est[:3, 3])

        pts_gt = (T_gt[:3, :3] @ self.mesh_points.T).T + T_gt[:3, 3]
        pts_est = (T_est[:3, :3] @ self.mesh_points.T).T + T_est[:3, 3]

        distances = np.linalg.norm(pts_gt - pts_est, axis=1)
        return float(np.mean(distances))

    def _compute_translation_error(
        self,
        T_gt: np.ndarray,
        T_est: np.ndarray
    ) -> float:
        """Compute Absolute Translation Error (ATE) in meters."""
        t_gt = T_gt[:3, 3]
        t_est = T_est[:3, 3]
        return float(np.linalg.norm(t_gt - t_est))

    def _compute_rotation_error(
        self,
        T_gt: np.ndarray,
        T_est: np.ndarray
    ) -> float:
        """Compute ARE, the absolute rotation error in degrees, from the relative rotation trace."""
        R_gt = T_gt[:3, :3]
        R_est = T_est[:3, :3]

        R_rel = R_gt.T @ R_est

        # clamp trace to avoid numerical errors with arccos
        trace = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
        angle_rad = np.arccos(trace)
        
        return float(np.degrees(angle_rad))

    def _pose_callback(self, msg: PoseStamped):
        """Receive estimated pose and compute metrics against ground truth."""
        self.frame_count += 1

        # Skip warmup frames while FoundationPose initializes
        if self.frame_count <= self.warmup_frames:
            return

        T_est = self._pose_to_matrix(msg)
        T_gt = self.ground_truth_pose

        add = self._compute_add(T_gt, T_est)
        ate = self._compute_translation_error(T_gt, T_est)
        are = self._compute_rotation_error(T_gt, T_est)

        result = {
            'frame': self.frame_count,
            'timestamp': time.time(),
            'add': add,
            'ate': ate,
            'are': are,
            'add_correct': add < 0.01  # Correct if ADD < 1cm
        }
        self.estimated_poses.append(result)

        if not self.benchmark_active:
            self.benchmark_active = True
            self.get_logger().info(
                "Warmup complete. Benchmark collection started!"
            )

        if (len(self.estimated_poses) >= self.num_samples
                and not self.results_saved):
            self._save_results()

    def _print_metrics(self):
        """Print live metrics summary every 5 seconds."""
        if len(self.estimated_poses) < 10:
            remaining = self.warmup_frames - self.frame_count
            if remaining > 0:
                self.get_logger().info(
                    f"Warming up... {remaining} frames remaining"
                )
            return

        adds = [r['add'] for r in self.estimated_poses]
        ates = [r['ate'] for r in self.estimated_poses]
        ares = [r['are'] for r in self.estimated_poses]
        correct = [r['add_correct'] for r in self.estimated_poses]

        # ADD-0.1d accuracy: percentage of frames with ADD < 1cm
        accuracy = sum(correct) / len(correct) * 100

        summary = (
            f"\n{'='*50}\n"
            f"  BENCHMARK: {len(self.estimated_poses)} frames\n"
            f"{'='*50}\n"
            f"  ADD mean:     {np.mean(adds)*100:.2f} cm\n"
            f"  ADD median:   {np.median(adds)*100:.2f} cm\n"
            f"  ADD std:      {np.std(adds)*100:.2f} cm\n"
            f"  ATE mean:     {np.mean(ates)*100:.2f} cm\n"
            f"  ARE mean:     {np.mean(ares):.2f} deg\n"
            f"  ADD-0.1d acc: {accuracy:.1f}%\n"
            f"{'='*50}"
        )

        self.get_logger().info(summary)

        msg = String()
        msg.data = (
            f"ADD={np.mean(adds)*100:.2f}cm "
            f"ATE={np.mean(ates)*100:.2f}cm "
            f"ARE={np.mean(ares):.2f}deg "
            f"ACC={accuracy:.1f}%"
        )
        self.metrics_pub.publish(msg)

    def _save_results(self):
        """Save benchmark results to a JSON file."""
        self.results_saved = True

        adds = [r['add'] for r in self.estimated_poses]
        ates = [r['ate'] for r in self.estimated_poses]
        ares = [r['are'] for r in self.estimated_poses]
        correct = [r['add_correct'] for r in self.estimated_poses]

        summary = {
            'object': 'YCB_004_sugar_box',
            'num_frames': len(self.estimated_poses),
            'add_mean_cm': float(np.mean(adds) * 100),
            'add_median_cm': float(np.median(adds) * 100),
            'add_std_cm': float(np.std(adds) * 100),
            'ate_mean_cm': float(np.mean(ates) * 100),
            'are_mean_deg': float(np.mean(ares)),
            'add_0_1d_accuracy_pct': float(
                sum(correct) / len(correct) * 100
            ),
            'raw_results': self.estimated_poses
        }

        results_path = os.path.join(
            self.results_dir, 'benchmark_results.json'
        )
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.get_logger().info(
            f"Results saved to {results_path}"
        )

        self.get_logger().info(
            f"\n{'='*50}\n"
            f"  FINAL BENCHMARK RESULTS\n"
            f"  Object: YCB 004 Sugar Box\n"
            f"  Frames: {len(self.estimated_poses)}\n"
            f"{'='*50}\n"
            f"  ADD mean:     {summary['add_mean_cm']:.2f} cm\n"
            f"  ADD median:   {summary['add_median_cm']:.2f} cm\n"
            f"  ATE mean:     {summary['ate_mean_cm']:.2f} cm\n"
            f"  ARE mean:     {summary['are_mean_deg']:.2f} deg\n"
            f"  ADD-0.1d:     {summary['add_0_1d_accuracy_pct']:.1f}%\n"
            f"{'='*50}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()