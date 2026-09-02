# 6-DoF Object Pose Estimation with FoundationPose + SAM2 (ROS2 Jazzy)

Most robot manipulation demos you see online rely on AprilTags, the little printed markers stuck to objects that tell the robot exactly where something is. That's fine for demos, but it doesn't work in the real world where objects don't come with stickers.

This project removes that dependency entirely. Point a depth camera at an object, give the system a 3D mesh of it, and it figures out the full 6-DoF pose (position and orientation in 3D space) with no markers, no retraining, and no per-object setup. That pose estimate then drives a UR5e arm through MoveIt2 to actually pick the object up.

The pipeline runs in ROS2 Jazzy with Gazebo simulation, and it's now closed end to end: camera in, robot picking things up out.

![Status](https://img.shields.io/badge/status-in--development-yellow)
![ROS2](https://img.shields.io/badge/ROS2-Jazzy-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## What it does

An RGB-D camera feed goes through two stages. First, SAM2 (Meta's segment anything model) automatically segments the target object from the scene, no bounding box needed from the user. That mask gets handed to FoundationPose (NVIDIA, 2024), which estimates the full 6-DoF pose and publishes it as a `geometry_msgs/PoseStamped` on a ROS2 topic, transformed into the world frame via TF2.

That pose feeds straight into MoveIt2, which plans a collision-free path to the object, closes a Robotiq 2F-85 gripper around it, lifts it, and returns to a safe home position, all without a hardcoded coordinate anywhere in the loop.

The whole thing runs in Gazebo Harmonic so you don't need physical hardware to develop or test it.

---
![RViz2 Visualization](docs/media/rviz2-visualization.png)
*Figure: Complete visualization pipeline in RViz2, showing the UR5e robot model, coordinate transforms (TF), estimated 6-DoF pose from FoundationPose, grasp markers, and SAM2 segmentation overlay*

## Why this matters

My earlier TIAGo pick-and-place project ([link](https://github.com/Sammykrishna/tiago-moveit2-pathplanning)) used AprilTags for object detection. It worked, but it's fundamentally limited: you need to know in advance which object you're grasping and stick a marker on it. Real industrial and humanoid robot applications don't have that luxury.

FoundationPose is zero-shot. You give it a mesh file once, and it can estimate the pose of that object in any scene, any lighting, any background. SAM2 handles the segmentation automatically so the pipeline requires zero human input at runtime.

---

## Current status

The full pipeline works end to end in simulation, from camera pixels to a completed pick-and-place cycle.

**Perception:**
- A simulated Intel RealSense D435i in Gazebo publishes synchronized RGB and depth streams.
- SAM2 automatically segments the target object every frame, with just an initial point hint, no manual bounding box.
- FoundationPose initializes a 6-DoF pose estimate from the SAM2 mask and depth data, then tracks the object in real time once initialized. Initialization now waits for a real SAM2 mask instead of falling back to a coarse depth-threshold heuristic, which used to cause the node to lock onto a stable but wrong pose without any obvious error.
- The estimated pose (in the camera's optical frame) gets transformed into the world frame using a proper ROS2 TF2 tree (`world -> camera_link -> camera_optical`) instead of hand-rolled transform matrices, so the math follows standard ROS conventions (REP-103).
- Position accuracy is around 2.5mm on X/Y against known ground truth. There's a small, fully explained offset on Z: the mesh's local origin sits at the object's bottom face rather than its center, so the raw pose describes where the bottom of the box touches the table. The manipulation side corrects for this (see below).
- The scene geometry matches physical reality. The table is a tabletop slab on four legs rather than a solid block down to the floor, and the robot base sits at table height beside the table edge, both necessary fixes since a solid-block table leaves no room for the arm and a floor-mounted base can't reach a tabletop object.
- Everything runs on a single consumer laptop GPU (RTX 4050, 6GB VRAM). Getting FoundationPose and SAM2 to fit in that memory budget together meant running SAM2 on CPU, shrinking FoundationPose's rotation candidate grid, running Gazebo headless, and tuning `PYTORCH_CUDA_ALLOC_CONF` to reduce fragmentation.

**Manipulation:**
- A Robotiq 2F-85 gripper is mounted on the UR5e via a proper adapter plate, with a dedicated TCP frame that already accounts for the gripper's physical reach so the perception team's raw pose doesn't need any manual offset tweaking downstream.
- A live MoveIt2 planning scene tracks both the table and the object as real collision geometry, updated from `/object_pose` in real time, not a fixed scene authored by hand.
- The grasp orientation is computed automatically from the object's estimated yaw, aligning the gripper's jaw axis with the object's shorter side. The target box's long side is within 2mm of the gripper's maximum opening, so getting this right isn't optional, a naive fixed approach angle would fail depending on how the object happened to be rotated on the table.
- The bottom-to-center correction mentioned above (perception reports the object's bottom face, not its center) is implemented as a single shared utility, used identically by both the planning scene's collision geometry and the grasp pose computation, so there's one source of truth instead of two numbers that could quietly drift apart.
- The full sequence (home, open gripper, approach, descend, close gripper, attach object, lift, return home, release) has been verified working end to end using live, camera-derived poses, not just hand-typed test coordinates.

---

## Results at a glance

| Area | Status | Detail |
|---|---|---|
| Perception accuracy (X/Y) | Done | ~2.5 mm against known ground truth |
| Zero-shot pose estimation | Done | No markers, no per-object training, mesh file only |
| GPU memory footprint | Done | Fits comfortably in 6 GB (RTX 4050 Mobile) |
| Live scene understanding for planning | Done | Table and object tracked as real MoveIt2 collision geometry, updated from perception in real time |
| Orientation-aware grasp planning | Done | Gripper jaw automatically aligned to the object's shorter axis from its estimated yaw |
| Full pick, lift, and return cycle | Done | Verified on live camera-derived poses, not just synthetic test coordinates |
| Planner robustness | In progress | Some grasp poses occasionally need a retry under the default planning time budget |
| Placing at an arbitrary target location | Not yet | Currently returns to and releases at a fixed home pose |
| Formal benchmark (YCB-Video, ADD metric) | Not yet | Planned |

A real pose sample from a live run, straight off `/object_pose`:

```
position:  x = 0.797,  y = 0.016,  z = 0.752
```

That Z value lands within about 2mm of the table's known surface height (0.750m), which is exactly what you'd expect for a box resting on the table with a small amount of physics-engine settling. It's a nice sanity check that the whole chain, camera through to a real-world-shaped number, is behaving.

---

## Manipulation in action

![Grasp approach, perception view](docs/media/rviz2-robot_picking.png)
*Figure: The UR5e approaching the object mid-sequence, viewed through the perception pipeline's own RViz2 window (SAM2 mask and camera feed panels on the left).*

![Grasp approach, planning view](docs/media/rviz2-planner_image.png)
*Figure: The same moment viewed through MoveIt2's RViz2 instance, showing the MotionPlanning panel and the live planning scene the pose estimate feeds into.*

---

## A few bugs along the way

Worth mentioning since they're the kind of thing that can bite anyone building a ROS2 plus simulation plus ML pipeline:

- A race condition where FoundationPose could initialize from a crude fallback mask if SAM2 hadn't published its first real segmentation yet. The resulting pose looked stable frame to frame, since tracking just refines from wherever it started, while actually being wrong the whole time. That made it look like a coordinate frame bug for a long time before the real cause turned up.
- Two different nodes were both publishing `world -> base_link` on `/tf_static`, our own static transform and one hardcoded inside the UR5e's stock URDF, so TF was flip-flopping between two different robot positions depending on which message arrived last. The same category of bug reappeared later when integrating the perception launch file with MoveIt2's own launch file, both starting a `robot_state_publisher` and `joint_state` source, and was fixed the same way: pick one authority, gate the other off.
- A 2x scale mismatch between the object mesh and its Gazebo collision geometry.
- A packaged gripper hardware macro that gave every mimic joint (the ones mechanically slaved to the gripper's main actuated joint) its own command interface, which `ros2_control`'s resource manager flatly refuses. Fixed by writing a minimal three-line hardware block that only claims the one real actuated joint.
- The gripper's touch links weren't fully enumerated for collision allowance during a grasp: it turned out the knuckle links, not just the fingers, are the ones that actually contact the object, found by reading the real error message closely rather than guessing.

---

## Stack

| Component | Tool | Version |
|-----------|------|---------|
| Robot middleware | ROS2 | Jazzy |
| Simulation | Gazebo Harmonic | - |
| Pose estimation | FoundationPose (NVIDIA) | 2024 |
| Segmentation | SAM2 (Meta) | 2024 |
| Motion planning | MoveIt2 | Jazzy |
| End effector | Robotiq 2F-85 | - |
| Language | Python | 3.10+ |
| Hardware req. | NVIDIA GPU | CUDA 12+ |

---

## Project structure

```
foundationpose-ros2/
├── ros2_ws/
│   └── src/
│       ├── simulation_pkg/
│       │   ├── worlds/                        # Gazebo world (table, camera, target object)
│       │   ├── launch/                        # Full pipeline launch file
│       │   └── rviz/                          # RViz2 config for visualization
│       ├── pose_estimation_pkg/
│       │   ├── pose_estimation_pkg/
│       │   │   ├── foundationpose_node.py     # FoundationPose ROS2 wrapper
│       │   │   └── sam2_node.py               # SAM2 segmentation node
│       │   ├── config/
│       │   │   └── params.yaml                # Camera topics, thresholds, mesh paths
│       │   └── meshes/                        # YCB object mesh files (.obj)
│       ├── robot_control_pkg/
│       │   ├── urdf/                          # UR5e + Robotiq 2F-85 wrapper xacro
│       │   └── robot_control_pkg/             # Grasp executor, planning scene manager, geometry utils
│       └── ur5e_robotiq_moveit_config/         # Generated MoveIt2 config (SRDF, controllers, kinematics)
├── FoundationPose/                             # NVlabs FoundationPose (submodule)
├── sam2/                                       # Meta SAM2 (submodule)
└── docs/
    └── media/                                  # Demo GIFs and screenshots
```

---

## Roadmap

- [x] Repository scaffold and project structure
- [x] Gazebo simulation with RGB-D camera (Intel RealSense D435i)
- [x] FoundationPose ROS2 node, subscribes to depth + color topics, publishes PoseStamped
- [x] SAM2 segmentation node, automatic object masking, no bounding box required
- [x] Full pipeline integration and visualization in RViz2
- [x] TF2-based camera-to-world pose transformation following REP-103 conventions
- [x] Calibration refinement: mesh/collision scale matched, SAM2 initialization race fixed, millimeter-level X/Y accuracy achieved
- [x] Scene geometry made physically realistic: tabletop on legs, reachable robot base placement
- [x] Robotiq 2F-85 gripper integrated with a dedicated TCP frame
- [x] Live MoveIt2 planning scene driven by the perception pipeline's object pose
- [x] Orientation-aware grasp planning, no hardcoded approach angle
- [x] Full pick, lift, and return-home cycle verified on live camera-derived poses
- [ ] Tune planner reliability so grasp poses solve consistently on the first attempt
- [ ] Place the object at an arbitrary target location instead of releasing at home
- [ ] Benchmark on YCB-Video objects, reporting ADD (Average Distance) metric

### What's next

With perception and manipulation both working and talking to each other, the remaining work is refinement rather than new plumbing: tightening up planner reliability so every grasp attempt solves on the first try instead of occasionally needing a retry, extending the pipeline to place the object somewhere other than back at home, and running a proper benchmark against YCB-Video to put a real number on pose accuracy across a range of objects rather than a single measured case.

---
## Getting started

### Requirements

- Ubuntu 24.04
- ROS2 Jazzy
- NVIDIA GPU with CUDA 12+
- Python 3.10+

### Build

```bash
git clone https://github.com/Sammykrishna/foundationpose-ros2.git
cd foundationpose-ros2/ros2_ws

colcon build
source install/setup.bash
```

Docker instructions will be added in the next commit once the simulation environment is set up.

---

## Background

This project is part of my M.Sc. Mechatronics studies at RWU Weingarten, building on earlier work in autonomous manipulation and sensor fusion. The goal is a robot arm that can pick up an arbitrary object from a table, no markers, no object-specific training, purely from a depth camera and a mesh file, and that goal is now demonstrated end to end in simulation.

---

## Author

**Samanth Krishna**
M.Sc. Mechatronics, Ravensburg-Weingarten University of Applied Sciences

[LinkedIn](https://linkedin.com/in/samanth-krishna-429126202) · [GitHub](https://github.com/Sammykrishna) · [Other projects](https://github.com/Sammykrishna?tab=repositories)