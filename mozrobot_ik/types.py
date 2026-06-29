#!/usr/bin/env python3
"""
基础数据类型 (无外部 SDK 依赖)
Copyright (c) 2025 Spirit AI. All rights reserved.
"""

from dataclasses import dataclass
from enum import Enum


class Arm(Enum):
    """目标手臂。value 即 URDF 中的关节/坐标系前缀。"""
    LEFT = "LeftArm"
    RIGHT = "RightArm"


@dataclass
class Pose:
    """末端位姿: 位置 (m) + 单位四元数。"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
