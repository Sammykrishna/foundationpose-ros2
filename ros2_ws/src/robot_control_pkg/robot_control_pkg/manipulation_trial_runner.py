#!/usr/bin/env python3
"""Automated verification harness for the manipulation loop (Milestone A).

Replaces "check RViz's Scene Objects panel by eye" with a programmatic
check against MoveIt2's own authoritative planning scene: it calls the
/get_planning_scene service to confirm sugar_box actually moves from the
world collision objects list to the robot's attached_collision_objects
list on attach, and back again on detach, never appearing in both at
once. It then drives N randomized grasp trials against a live
grasp_executor + planning_scene_manager + MoveIt2 stack and reports a
real success rate.

Run against a live stack with synthetic poses (manipulation loop only):
    ros2 launch ur5e_robotiq_moveit_config demo.launch.py use_rviz:=false
    ros2 run robot_control_pkg planning_scene_manager
    ros2 run robot_control_pkg grasp_executor
    ros2 run robot_control_pkg manipulation_trial_runner --ros-args -p num_trials:=20

Run with passive_mode:=true for a true end-to-end measurement instead:
no synthetic poses are published, grasp_executor triggers only when SAM2 +
FoundationPose actually produce a stable real pose, driving both perception
and manipulation for real:
    ros2 launch simulation_pkg gazebo.launch.py standalone:=false
    ros2 launch ur5e_robotiq_moveit_config demo.launch.py use_rviz:=false
    ros2 run robot_control_pkg planning_scene_manager
    ros2 run robot_control_pkg grasp_executor
    ros2 run robot_control_pkg manipulation_trial_runner --ros-args \
        -p passive_mode:=true -p num_trials:=8
"""

import math
import random
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import PlanningSceneComponents

BOX_ID = 'sugar_box'
TERMINAL_STATES = {'DONE', 'ERROR'}

# Table spans x:[0.2,1.4], y:[-0.4,0.4] (TABLE_TOP_POSE=(0.8,0,0.725),
# size=(1.2,0.8,0.05)); base sits at (0.8,-0.45,0.752). Inset from the
# table edges so randomized poses stay kinematically reachable.
X_RANGE = (0.65, 0.95)
Y_RANGE = (-0.15, 0.25)
BOX_BOTTOM_Z = 0.752  # table surface height, matches a box resting on it


def random_box_pose(rng: random.Random, frame_id: str = 'world') -> PoseStamped:
    x = rng.uniform(*X_RANGE)
    y = rng.uniform(*Y_RANGE)
    yaw = rng.uniform(-math.pi, math.pi)

    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = BOX_BOTTOM_Z
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.orientation.w = math.cos(yaw / 2.0)
    return msg


class TrialRunner(Node):
    def __init__(self):
        super().__init__('manipulation_trial_runner')

        self.declare_parameter('num_trials', 10)
        self.declare_parameter('seed', 42)
        self.declare_parameter('settle_publishes', 15)
        self.declare_parameter('trial_timeout_sec', 90.0)
        self.declare_parameter('poll_period_sec', 0.3)
        self.declare_parameter('report_path', '/tmp/manipulation_trial_report.txt')
        self.declare_parameter('passive_mode', False)
        self.declare_parameter('perception_wait_timeout_sec', 120.0)

        self.num_trials = self.get_parameter('num_trials').value
        self.seed = self.get_parameter('seed').value
        self.settle_publishes = self.get_parameter('settle_publishes').value
        self.trial_timeout = self.get_parameter('trial_timeout_sec').value
        self.poll_period = self.get_parameter('poll_period_sec').value
        self.report_path = self.get_parameter('report_path').value
        self.passive_mode = self.get_parameter('passive_mode').value
        self.perception_wait_timeout = self.get_parameter('perception_wait_timeout_sec').value

        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)
        self.status = 'IDLE'
        self.create_subscription(String, '/grasp_status', self._status_cb, 10)

        self.latest_real_pose = None
        if self.passive_mode:
            self.create_subscription(PoseStamped, '/object_pose', self._real_pose_cb, 10)

        self.gps_client = self.create_client(GetPlanningScene, '/get_planning_scene')

        self.rng = random.Random(self.seed)
        self.results = []

    def _status_cb(self, msg: String):
        self.status = msg.data

    def _real_pose_cb(self, msg: PoseStamped):
        self.latest_real_pose = msg

    def _spin_for(self, seconds: float):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=max(0.0, deadline - time.monotonic()))

    def _wait_until(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate():
                return True
        return False

    def query_scene(self):
        """Ground-truth check via MoveIt2's own planning scene, not RViz."""
        if not self.gps_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/get_planning_scene service unavailable")
            return None, None
        req = GetPlanningScene.Request()
        req.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )
        future = self.gps_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            return None, None
        scene = future.result().scene
        world_names = {obj.id for obj in scene.world.collision_objects}
        attached_names = {a.object.id for a in scene.robot_state.attached_collision_objects}
        return world_names, attached_names

    def run_trial(self, idx: int) -> dict:
        result = {
            'trial': idx,
            'pose': None,
            'final_status': None,
            'saw_clean_attach': False,   # attached, and NOT also in world
            'saw_duplication': False,    # in world AND attached at once
            'end_state_clean': None,     # box back in world, not attached, after the trial
        }

        if not self._wait_until(lambda: self.status == 'IDLE', timeout=30.0):
            self.get_logger().error(f"Trial {idx}: executor never returned to IDLE, skipping")
            result['final_status'] = 'STUCK_NOT_IDLE'
            return result

        if self.passive_mode:
            # Don't publish anything: wait for SAM2 + FoundationPose to
            # produce a real stable pose and let grasp_executor trigger
            # on its own, exactly as it would with a live camera.
            if not self._wait_until(lambda: self.status != 'IDLE',
                                     timeout=self.perception_wait_timeout):
                self.get_logger().error(
                    f"Trial {idx}: perception never produced a stable pose "
                    f"within {self.perception_wait_timeout}s"
                )
                result['final_status'] = 'NO_STABLE_PERCEPTION'
                return result
            if self.latest_real_pose is not None:
                p = self.latest_real_pose.pose
                result['pose'] = (
                    p.position.x, p.position.y,
                    2.0 * math.atan2(p.orientation.z, p.orientation.w),
                )
        else:
            pose = random_box_pose(self.rng)
            result['pose'] = (
                pose.pose.position.x, pose.pose.position.y,
                2.0 * math.atan2(pose.pose.orientation.z, pose.pose.orientation.w),
            )

            for _ in range(self.settle_publishes):
                pose.header.stamp = self.get_clock().now().to_msg()
                self.pose_pub.publish(pose)
                self._spin_for(0.1)

        deadline = time.monotonic() + self.trial_timeout
        while time.monotonic() < deadline and self.status not in TERMINAL_STATES:
            rclpy.spin_once(self, timeout_sec=self.poll_period)
            if not self.passive_mode:
                # keep feeding the pose so grasp_executor's stability buffer
                # (and planning_scene_manager's live box) doesn't go stale
                pose.header.stamp = self.get_clock().now().to_msg()
                self.pose_pub.publish(pose)

            world_names, attached_names = self.query_scene()
            if world_names is None:
                continue
            in_world = BOX_ID in world_names
            in_attached = BOX_ID in attached_names
            if in_world and in_attached:
                result['saw_duplication'] = True
            elif in_attached and not in_world:
                result['saw_clean_attach'] = True

        result['final_status'] = self.status if self.status in TERMINAL_STATES else 'TIMEOUT'

        # give grasp_executor's own detach (or the ERROR-path recovery
        # detach) a moment to land, then check end-of-trial scene state
        self._spin_for(1.0)
        world_names, attached_names = self.query_scene()
        if world_names is not None:
            in_world = BOX_ID in world_names
            in_attached = BOX_ID in attached_names
            result['end_state_clean'] = in_world and not in_attached
            if in_world and in_attached:
                result['saw_duplication'] = True

        # let grasp_executor's 5s reset timer return it to IDLE before the next trial
        self._wait_until(lambda: self.status == 'IDLE', timeout=10.0)
        return result

    def run(self):
        self.get_logger().info(
            f"Running {self.num_trials} manipulation trials (seed={self.seed})..."
        )
        for i in range(self.num_trials):
            r = self.run_trial(i)
            self.results.append(r)
            self.get_logger().info(f"Trial {i}: {r}")
        self.report()

    def report(self):
        n = len(self.results)
        successes = sum(1 for r in self.results if r['final_status'] == 'DONE')
        clean_attach = sum(1 for r in self.results if r['saw_clean_attach'])
        duplications = sum(1 for r in self.results if r['saw_duplication'])
        clean_end = sum(1 for r in self.results if r['end_state_clean'])

        mode = "PASSIVE / REAL PERCEPTION (SAM2 + FoundationPose)" if self.passive_mode \
            else "SYNTHETIC POSES (manipulation loop only)"
        lines = []
        lines.append("=" * 60)
        lines.append(f"MANIPULATION TRIAL REPORT — {mode}")
        lines.append("=" * 60)
        lines.append(f"Trials run:                         {n}")
        lines.append(f"Grasp sequence successes (DONE):    {successes}/{n} "
                      f"({100.0 * successes / n:.1f}%)" if n else "n/a")
        lines.append(f"Clean attach observed (no dup):     {clean_attach}/{n}")
        lines.append(f"Duplication observed (world+attach):{duplications}/{n}")
        lines.append(f"Clean end state (box back in world):{clean_end}/{n}")
        lines.append("=" * 60)
        for r in self.results:
            lines.append(str(r))
        report_text = "\n".join(lines)
        self.get_logger().info("\n" + report_text)

        with open(self.report_path, 'w') as f:
            f.write(report_text + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = TrialRunner()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
