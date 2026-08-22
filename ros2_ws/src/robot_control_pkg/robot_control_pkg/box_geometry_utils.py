#!/usr/bin/env python3
"""
Shared utility for correcting FoundationPose's bottom-center pose
convention into the true geometric center that MoveIt2 collision
objects and grasp targets need.

FoundationPose reports /object_pose using the sugar box mesh's local
origin, which sits at the bottom face center, not the box's actual
center (see pipeline handoff, section 2.5). The correction is
rotation-aware: if the box is yawed on the table, "up" in the box's
own frame is not the same as world-frame +Z.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose # type: ignore

# Real YCB sugar box dimensions (X x Y x Z), matching table_scene.sdf
SUGAR_BOX_SIZE = (0.0495, 0.0942, 0.176)
SUGAR_BOX_HALF_HEIGHT = SUGAR_BOX_SIZE[2] / 2.0


def get_box_center_from_bottom_pose(bottom_pose: Pose, half_height: float) -> Pose:
    """Given a Pose at the box's bottom-face-center, return the Pose at
    its true geometric center, accounting for current orientation."""
    quat_xyzw = [
        bottom_pose.orientation.x,
        bottom_pose.orientation.y,
        bottom_pose.orientation.z,
        bottom_pose.orientation.w,
    ]
    rot = Rotation.from_quat(quat_xyzw)
    world_offset = rot.apply(np.array([0.0, 0.0, half_height]))

    center_pose = Pose()
    center_pose.position.x = bottom_pose.position.x + world_offset[0]
    center_pose.position.y = bottom_pose.position.y + world_offset[1]
    center_pose.position.z = bottom_pose.position.z + world_offset[2]
    center_pose.orientation = bottom_pose.orientation
    return center_pose