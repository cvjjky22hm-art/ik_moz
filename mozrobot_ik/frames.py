#!/usr/bin/env python3
"""
坐标系工具 + HMD 追踪数据 Z 重锚定

- Pose <-> pinocchio.SE3 转换
- world(全局) <-> base_link 位姿变换
- GlobalPoseRebaser: Meta Quest 等头显地面平面不可靠, 将首帧 Z 重置为
  躯干标称高度 (默认 1.2 m), 之后按帧间 Z 增量累加得到实时躯干高度。

Copyright (c) 2025 Spirit AI. All rights reserved.
"""

import numpy as np
import pinocchio

from .types import Pose


def pose_to_se3(p: Pose):
    """Pose -> pinocchio.SE3"""
    quat = pinocchio.Quaternion(p.qw, p.qx, p.qy, p.qz)
    quat.normalize()
    return pinocchio.SE3(quat.matrix(), np.array([p.x, p.y, p.z]))


def se3_to_pose(T) -> Pose:
    """pinocchio.SE3 -> Pose"""
    t = T.translation
    quat = pinocchio.Quaternion(T.rotation)
    quat.normalize()
    return Pose(
        x=float(t[0]), y=float(t[1]), z=float(t[2]),
        qx=float(quat.x), qy=float(quat.y), qz=float(quat.z), qw=float(quat.w),
    )


def world_to_base(pose_world: Pose, base_in_world: Pose) -> Pose:
    """全局系位姿 -> base_link 系位姿:  T_base_target = T_wb⁻¹ · T_wt

    :param pose_world:    目标在全局(世界)系下的位姿
    :param base_in_world: 机器人 base_link 在全局系下的位姿 (如里程计)
    """
    T_wb = pose_to_se3(base_in_world)
    T_wt = pose_to_se3(pose_world)
    return se3_to_pose(T_wb.actInv(T_wt))


def base_to_world(pose_base: Pose, base_in_world: Pose) -> Pose:
    """base_link 系位姿 -> 全局系位姿:  T_wt = T_wb · T_bt"""
    T_wb = pose_to_se3(base_in_world)
    T_bt = pose_to_se3(pose_base)
    return se3_to_pose(T_wb.act(T_bt))


class GlobalPoseRebaser:
    """HMD 追踪位姿 Z 重锚定 (躯干高度)。

    头显输出的地面平面不准, 绝对 Z 不可信。处理方法:
    首帧 Z 重置为 z0 (躯干标称高度 1.2 m), 之后逐帧累加 Z 增量
    (z_out += z_k - z_{k-1}) 得到实时躯干高度。

    典型用法 (遥操作循环):
        rb = GlobalPoseRebaser()          # z0=1.2
        pose = rb.update(hmd_pose)        # 每帧调用
        rb.reset()                        # 操作员重新对中时
    """

    def __init__(self, z0: float = 1.2, max_dz_per_frame: float = None,
                 rebase_xy: bool = False):
        """
        :param z0:               首帧锚定高度 (m), 默认 1.2
        :param max_dz_per_frame: 单帧 Z 增量限幅 (m); None 不限幅。
                                 用于抑制头显跟踪跳变。
        :param rebase_xy:        True 时 X/Y 也改为相对首帧 (首帧 XY -> 0)
        """
        self.z0 = z0
        self.max_dz_per_frame = max_dz_per_frame
        self.rebase_xy = rebase_xy
        self.reset()

    def reset(self):
        """清除锚定; 下一帧重新作为首帧 (Z 重置为 z0)。"""
        self._z_prev_raw = None
        self._z_out = None
        self._xy_ref = None

    def update(self, pose_raw: Pose) -> Pose:
        """输入一帧原始追踪位姿, 返回 Z 重锚定后的位姿。"""
        if self._z_prev_raw is None:
            self._z_out = self.z0
            self._xy_ref = (pose_raw.x, pose_raw.y)
        else:
            dz = pose_raw.z - self._z_prev_raw
            if self.max_dz_per_frame is not None:
                lim = abs(self.max_dz_per_frame)
                dz = max(-lim, min(lim, dz))
            self._z_out += dz
        self._z_prev_raw = pose_raw.z

        x, y = pose_raw.x, pose_raw.y
        if self.rebase_xy:
            x -= self._xy_ref[0]
            y -= self._xy_ref[1]
        return Pose(x=x, y=y, z=self._z_out,
                    qx=pose_raw.qx, qy=pose_raw.qy,
                    qz=pose_raw.qz, qw=pose_raw.qw)
