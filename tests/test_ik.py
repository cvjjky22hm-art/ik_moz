#!/usr/bin/env python3
"""
离线单元测试 (pytest)
    pip install pytest && pytest
Copyright (c) 2025 Spirit AI. All rights reserved.
"""

import math

from mozrobot_ik import ArmIK, IKOptions, Arm, get_spirit01_urdf

LEFT_ARM_WORK_POINT_DEG = [-9, -50, -20, -90, -35, 8, -7]


def _make_ik(arm=Arm.LEFT):
    return ArmIK(get_spirit01_urdf(), arm)


def test_model_dof():
    ik = _make_ik()
    assert ik.dof() == 7
    assert ik.ee_frame_name == "LeftArm_ee"


def test_fk_ik_roundtrip():
    """FK 得到目标, 从零种子 IK 应收敛回该位姿。"""
    ik = _make_ik()
    q_true = [math.radians(d) for d in LEFT_ARM_WORK_POINT_DEG]
    target = ik.fk(q_true)
    ok, q_out = ik.solve(target, [0.0] * ik.dof())
    assert ok
    assert ik.pose_error(target, q_out) < 1e-3


def test_psi_redundancy():
    """同一末端位姿下, 臂角应被驱动到目标且末端保持。"""
    ik = _make_ik()
    q_true = [math.radians(d) for d in LEFT_ARM_WORK_POINT_DEG]
    target = ik.fk(q_true)
    _, q0 = ik.solve(target, [0.0] * ik.dof())

    psi0 = ik.arm_angle(q0)
    opt = IKOptions()
    opt.use_psi = True
    opt.psi_target = psi0 + math.radians(20.0)
    ok, q_psi = ik.solve(target, q0, opt)

    assert ok
    assert ik.pose_error(target, q_psi) < 1e-3                 # 末端位姿保持
    assert abs(ik.arm_angle(q_psi) - opt.psi_target) < math.radians(5.0)  # 臂角到达


def test_right_arm():
    ik = _make_ik(Arm.RIGHT)
    assert ik.dof() == 7
    assert ik.ee_frame_name == "RightArm_ee"
    q = [math.radians(d) for d in [9, -50, 20, 90, 35, 8, 7]]
    target = ik.fk(q)
    ok, q_out = ik.solve(target, [0.0] * ik.dof())
    assert ok
    assert ik.pose_error(target, q_out) < 1e-3
