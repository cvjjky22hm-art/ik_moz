#!/usr/bin/env python3
"""
mozrobot_ik — 本地笛卡尔→关节逆运动学 (Pinocchio)

Copyright (c) 2025 Spirit AI. All rights reserved.
"""

from .types import Arm, Pose
from .solver import ArmIK, IKOptions, WholeBodyIK, WholeBodyOptions
from .frames import (GlobalPoseRebaser, base_to_world, pose_to_se3,
                     se3_to_pose, world_to_base)
from .urdf_data import get_spirit01_urdf, load_urdf

__version__ = "0.2.0"

__all__ = [
    "Arm",
    "Pose",
    "ArmIK",
    "IKOptions",
    "WholeBodyIK",
    "WholeBodyOptions",
    "GlobalPoseRebaser",
    "world_to_base",
    "base_to_world",
    "pose_to_se3",
    "se3_to_pose",
    "get_spirit01_urdf",
    "load_urdf",
    "__version__",
]
