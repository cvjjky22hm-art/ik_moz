#!/usr/bin/env python3
"""
mozrobot_ik — 本地笛卡尔→关节逆运动学 (Pinocchio)

Copyright (c) 2025 Spirit AI. All rights reserved.
"""

from .types import Arm, Pose
from .solver import ArmIK, IKOptions
from .urdf_data import get_spirit01_urdf, load_urdf

__version__ = "0.1.0"

__all__ = [
    "Arm",
    "Pose",
    "ArmIK",
    "IKOptions",
    "get_spirit01_urdf",
    "load_urdf",
    "__version__",
]
