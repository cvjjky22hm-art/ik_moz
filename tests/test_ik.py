#!/usr/bin/env python3
"""
离线单元测试 (pytest)
    pip install pytest && pytest
Copyright (c) 2025 Spirit AI. All rights reserved.
"""

import math

from mozrobot_ik import (Arm, ArmIK, GlobalPoseRebaser, IKOptions, Pose,
                         WholeBodyIK, WholeBodyOptions, base_to_world,
                         get_spirit01_urdf, world_to_base)

LEFT_ARM_WORK_POINT_DEG = [-9, -50, -20, -90, -35, 8, -7]
WAIST_BENT = [0.1, 0.2, -0.3, 0.2, 0.1, 0.2]  # 腰部弯曲测试配置 (限位内)


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


# ---------------- 全局(世界)系 API ----------------

BASE_IN_WORLD = Pose(x=1.0, y=2.0, z=0.3,
                     qx=0.0, qy=0.0, qz=math.sin(math.radians(15)),
                     qw=math.cos(math.radians(15)))  # yaw 30°


def test_world_frame_roundtrip():
    """fk_world / solve_world 往返: 全局系目标应收敛且与 fk_world 一致。"""
    ik = _make_ik()
    q_true = [math.radians(d) for d in LEFT_ARM_WORK_POINT_DEG]
    target_w = ik.fk_world(q_true, BASE_IN_WORLD)

    ok, q_out = ik.solve_world(target_w, BASE_IN_WORLD, [0.0] * ik.dof())
    assert ok
    assert ik.pose_error(world_to_base(target_w, BASE_IN_WORLD), q_out) < 1e-3

    got_w = ik.fk_world(q_out, BASE_IN_WORLD)
    assert abs(got_w.x - target_w.x) < 1e-3
    assert abs(got_w.y - target_w.y) < 1e-3
    assert abs(got_w.z - target_w.z) < 1e-3


def test_world_base_transform_inverse():
    """world_to_base 与 base_to_world 互逆。"""
    n = math.sqrt(0.1 ** 2 + 0.2 ** 2 + 0.3 ** 2 + 0.9 ** 2)
    p_w = Pose(x=0.5, y=-0.2, z=1.4,
               qx=0.1 / n, qy=0.2 / n, qz=0.3 / n, qw=0.9 / n)
    p_b = world_to_base(p_w, BASE_IN_WORLD)
    p_w2 = base_to_world(p_b, BASE_IN_WORLD)
    for attr in ("x", "y", "z", "qx", "qy", "qz", "qw"):
        assert abs(getattr(p_w2, attr) - getattr(p_w, attr)) < 1e-9


# ---------------- HMD Z 重锚定 ----------------

def test_rebaser():
    """首帧 Z -> 1.2, 之后按帧间增量累加; 限幅与 reset 生效。"""
    rb = GlobalPoseRebaser(z0=1.2, max_dz_per_frame=0.1)

    p1 = rb.update(Pose(x=0.5, y=0.6, z=0.37))
    assert abs(p1.z - 1.2) < 1e-12
    assert abs(p1.x - 0.5) < 1e-12  # XY 默认透传

    p2 = rb.update(Pose(z=0.40))            # +0.03
    assert abs(p2.z - 1.23) < 1e-12

    p3 = rb.update(Pose(z=0.90))            # 跳变 +0.5 -> 限幅 +0.1
    assert abs(p3.z - 1.33) < 1e-12

    rb.reset()
    p4 = rb.update(Pose(z=0.55))            # 重新锚定
    assert abs(p4.z - 1.2) < 1e-12


def test_rebaser_xy():
    """rebase_xy=True 时 XY 相对首帧。"""
    rb = GlobalPoseRebaser(rebase_xy=True)
    rb.update(Pose(x=2.0, y=-1.0, z=0.3))
    p = rb.update(Pose(x=2.1, y=-0.8, z=0.3))
    assert abs(p.x - 0.1) < 1e-12
    assert abs(p.y - 0.2) < 1e-12


# ---------------- 腰部配置更新 (set_waist_q / set_locked_q) ----------------

def test_set_waist_q():
    """弯腰后 fk 应与全身模型一致 (用 WholeBodyIK.fk 交叉验证), 且可求解。"""
    q_arm = [math.radians(d) for d in LEFT_ARM_WORK_POINT_DEG]

    ik = _make_ik()
    pose_neutral = ik.fk(q_arm)
    ik.set_waist_q(WAIST_BENT)
    pose_bent = ik.fk(q_arm)

    # 弯腰后末端位姿应明显不同
    d = math.sqrt((pose_bent.x - pose_neutral.x) ** 2 +
                  (pose_bent.y - pose_neutral.y) ** 2 +
                  (pose_bent.z - pose_neutral.z) ** 2)
    assert d > 0.05

    # 与 13-DOF 全身模型 FK 交叉验证
    wb = WholeBodyIK(get_spirit01_urdf(), Arm.LEFT)
    pose_wb = wb.fk(list(WAIST_BENT) + q_arm)
    for attr in ("x", "y", "z"):
        assert abs(getattr(pose_wb, attr) - getattr(pose_bent, attr)) < 1e-9

    # 弯腰模型上仍可正常求逆
    ok, q_out = ik.solve(pose_bent, [0.0] * ik.dof())
    assert ok
    assert ik.pose_error(pose_bent, q_out) < 1e-3


# ---------------- 全身 (腰+臂 13-DOF) IK ----------------

def test_wholebody_roundtrip():
    """13-DOF FK 目标, 从中性种子求解应收敛。"""
    wb = WholeBodyIK(get_spirit01_urdf(), Arm.LEFT)
    assert wb.dof() == 13
    q_true = list(WAIST_BENT) + \
        [math.radians(d) for d in LEFT_ARM_WORK_POINT_DEG]
    target = wb.fk(q_true)

    ok, q_out = wb.solve(target, [0.0] * wb.dof())
    assert ok
    assert wb.pose_error(target, q_out) < 1e-3

    q_waist, q_arm = wb.split_q(q_out)
    assert len(q_waist) == 6 and len(q_arm) == 7


def test_wholebody_torso_task():
    """零空间躯干任务: 末端位姿保持, 胸部高度向目标移动。"""
    wb = WholeBodyIK(get_spirit01_urdf(), Arm.LEFT)
    q0 = list(WAIST_BENT) + \
        [math.radians(d) for d in LEFT_ARM_WORK_POINT_DEG]
    target = wb.fk(q0)
    torso0 = wb.fk_torso(q0)

    opt = WholeBodyOptions()
    opt.use_torso_task = True
    opt.torso_target = Pose(x=torso0.x, y=torso0.y, z=torso0.z - 0.03)

    ok, q_out = wb.solve(target, q0, opt)
    assert ok
    assert wb.pose_error(target, q_out) < 1e-3          # 末端保持
    torso1 = wb.fk_torso(q_out)
    err0 = abs(torso0.z - opt.torso_target.z)           # 初始 0.03
    err1 = abs(torso1.z - opt.torso_target.z)
    assert err1 < err0 * 0.5                            # 明显向目标移动
