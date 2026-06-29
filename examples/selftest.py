#!/usr/bin/env python3
"""
离线自检: 验证 FK/IK 往返 + 臂角 psi 控制 (无需机器人)

用法:
    python examples/selftest.py
Copyright (c) 2025 Spirit AI. All rights reserved.
"""

import math
import sys

from mozrobot_ik import ArmIK, IKOptions, Arm, get_spirit01_urdf

# 左臂工作点位 (deg)
LEFT_ARM_WORK_POINT_DEG = [-9, -50, -20, -90, -35, 8, -7]


def fmt(q):
    return "[" + ", ".join(f"{v:.4f}" for v in q) + "]"


def print_pose(tag, p):
    print(f"{tag} pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) "
          f"quat=({p.qx:.4f}, {p.qy:.4f}, {p.qz:.4f}, {p.qw:.4f})")


def main():
    print("=== mozrobot-ik self-test (offline) ===")
    ik = ArmIK(get_spirit01_urdf(), Arm.LEFT)
    print(f"Arm DOF={ik.dof()}, ee_frame={ik.ee_frame_name}")

    # 1. 用工作点位作为"真值"关节角, FK 得到目标位姿
    q_true = [math.radians(d) for d in LEFT_ARM_WORK_POINT_DEG]
    target = ik.fk(q_true)
    print(f"q_true    {fmt(q_true)}")
    print_pose("FK(q_true) ->", target)

    # 2. 用偏离的种子做 IK
    ok, q_out = ik.solve(target, [0.0] * ik.dof())
    res = ik.pose_error(target, q_out)
    print(f"q_ik      {fmt(q_out)}")
    print(f"IK converged={ok}, residual={res:.3e}")
    print_pose("FK(q_ik)  ->", ik.fk(q_out))
    pass_pose = ok and res < 1e-3
    print("[PASS] pose IK" if pass_pose else "[FAIL] pose IK")

    # 3. 臂角 psi 冗余控制: 同一末端位姿, 把臂角驱动到不同目标
    print("--- arm-angle (psi) redundancy ---")
    psi_now = ik.arm_angle(q_out)
    print(f"current psi = {math.degrees(psi_now):.1f} deg")

    opt = IKOptions()
    opt.use_psi = True
    opt.psi_gain = 1.0
    opt.psi_target = psi_now + math.radians(20.0)
    ok2, q_psi = ik.solve(target, q_out, opt)
    res2 = ik.pose_error(target, q_psi)
    psi2 = ik.arm_angle(q_psi)
    print(f"q_ik(psi) {fmt(q_psi)}")
    print(f"converged={ok2}, pose_residual={res2:.3e}, "
          f"psi={math.degrees(psi2):.1f} deg (target {math.degrees(opt.psi_target):.1f})")
    pass_psi = ok2 and res2 < 1e-3 and abs(psi2 - opt.psi_target) < math.radians(5.0)
    print("[PASS] psi control" if pass_psi else "[WARN] psi not fully reached")

    return 0 if pass_pose else 1


if __name__ == "__main__":
    sys.exit(main())
