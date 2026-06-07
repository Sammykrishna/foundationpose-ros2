# 6-DoF Object Pose Estimation with FoundationPose + SAM2 (ROS2 Jazzy)

Most robot manipulation demos you see online rely on AprilTags little printed markers stuck to objects that tell the robot exactly where something is. That's fine for demos, but it doesn't work in the real world where objects don't come with stickers.

This project removes that dependency entirely. Point a depth camera at an object, give the system a 3D mesh of it, and it figures out the full 6-DoF pose position and orientation in 3D space without any markers, without retraining, without any setup per object.

The pipeline runs in ROS2 Jazzy with Gazebo simulation and is designed to feed directly into MoveIt2 for actual robot grasping.

![Status](https://img.shields.io/badge/status-in--development-yellow)
![ROS2](https://img.shields.io/badge/ROS2-Jazzy-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## What it does

An RGB-D camera feed goes through two stages. First, SAM2 (Meta's segment anything model) automatically segments the target object from the scene, no bounding box needed from the user. That mask gets handed to FoundationPose (NVIDIA, 2024), which estimates the full 6-DoF pose and publishes it as a `geometry_msgs/PoseStamped` on a ROS2 topic. MoveIt2 picks that up and plans a collision-free grasp.

The whole thing runs in Gazebo Harmonic so you don't need physical hardware to develop or test it.

---

## Why this matters

My earlier TIAGo pick-and-place project ([link](https://github.com/Sammykrishna/tiago-moveit2-pathplanning)) used AprilTags for object detection. It worked, but it's fundamentally limited, you need to know in advance which object you're grasping and stick a marker on it. Real industrial and humanoid robot applications don't have that luxury.

FoundationPose is zero-shot: you give it a mesh file once, and it can estimate the pose of that object in any scene, any lighting, any background. SAM2 handles the segmentation automatically so the pipeline requires zero human input at runtime.

---

## Stack

| Component | Tool | Version |
|-----------|------|---------|
| Robot middleware | ROS2 | Jazzy |
| Simulation | Gazebo Harmonic | — |
| Pose estimation | FoundationPose (NVIDIA) | 2024 |
| Segmentation | SAM2 (Meta) | 2024 |
| Motion planning | MoveIt2 | Jazzy |
| Language | Python | 3.10+ |
| Hardware req. | NVIDIA GPU | CUDA 12+ |

---

## Project structure

foundationpose-ros2/
├── ros2_ws/
│   └── src/
│       └── pose_estimation_pkg/
│           ├── pose_estimation_pkg/
│           │   ├── foundationpose_node.py     # FoundationPose ROS2 wrapper
│           │   └── sam2_node.py               # SAM2 segmentation node
│           ├── launch/
│           │   └── pose_estimation.launch.py  # Launches full pipeline
│           ├── config/
│           │   └── params.yaml                # Camera topics, thresholds, mesh paths
│           ├── meshes/                        # YCB object mesh files (.obj)
│           └── rviz/                          # RViz2 config for visualization
├── scripts/
│   └── run_demo.sh                            # One-command demo launcher
└── docs/
└── media/                                 # Demo GIFs and screenshots

---

## Roadmap

- [x] Repository scaffold and project structure
- [x] Gazebo simulation with RGB-D camera (Intel RealSense D435i)
- [x] FoundationPose ROS2 node — subscribes to depth + color topics, publishes PoseStamped
- [x] SAM2 segmentation node — automatic object masking, no bounding box required
- [x] Full pipeline integration and visualization in RViz2
- [x] MoveIt2 grasp execution using estimated pose
- [ ] Benchmark on YCB-Video objects — reporting ADD (Average Distance) metric
- [ ] Demo video and results

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

This project is part of my M.Sc. Mechatronics studies at RWU Weingarten, building on earlier work in autonomous manipulation and sensor fusion. The goal is to get to a point where a robot arm can pick up an arbitrary object from a table, no markers, no object-specific training, purely from a depth camera and a mesh file.

---

## Author

**Samanth Krishna**
M.Sc. Mechatronics — Ravensburg-Weingarten University of Applied Sciences

[LinkedIn](https://linkedin.com/in/samanth-krishna-429126202) · [GitHub](https://github.com/Sammykrishna) · [Other projects](https://github.com/Sammykrishna?tab=repositories)
