#!/usr/bin/env python3
"""
命令行入口
Copyright (c) 2025 Spirit AI. All rights reserved.
"""

import math
import sys

from .frames import GlobalPoseRebaser, world_to_base
from .solver import ArmIK, IKOptions, WholeBodyIK, WholeBodyOptions
from .types import Arm, Pose
from .urdf_data import get_spirit01_urdf

_LEFT_ARM_WORK_POINT_DEG = [-9, -50, -20, -90, -35, 8, -7]
_WAIST_BENT = [0.1, 0.2, -0.3, 0.2, 0.1, 0.2]


def selftest() -> int:
    """离线自检: FK/IK 往返 + psi 控制 + 全局系 + 重锚定 + 全身 IK (无需机器人)。"""
    print("=== mozrobot-ik self-test (offline) ===")
    urdf = get_spirit01_urdf()
    ik = ArmIK(urdf, Arm.LEFT)
    print(f"Arm DOF={ik.dof()}, ee_frame={ik.ee_frame_name}")

    q_true = [math.radians(d) for d in _LEFT_ARM_WORK_POINT_DEG]
    target = ik.fk(q_true)
    ok, q_out = ik.solve(target, [0.0] * ik.dof())
    res = ik.pose_error(target, q_out)
    print(f"pose IK: converged={ok}, residual={res:.3e} "
          f"-> {'[PASS]' if ok and res < 1e-3 else '[FAIL]'}")

    psi0 = ik.arm_angle(q_out)
    opt = IKOptions()
    opt.use_psi = True
    opt.psi_target = psi0 + math.radians(20.0)
    ok2, q_psi = ik.solve(target, q_out, opt)
    res2 = ik.pose_error(target, q_psi)
    psi2 = ik.arm_angle(q_psi)
    psi_pass = ok2 and res2 < 1e-3 and abs(psi2 - opt.psi_target) < math.radians(5.0)
    print(f"psi control: psi {math.degrees(psi0):.1f} -> {math.degrees(psi2):.1f} deg "
          f"(target {math.degrees(opt.psi_target):.1f}), pose_res={res2:.3e} "
          f"-> {'[PASS]' if psi_pass else '[WARN]'}")

    # ---- 全局(世界)系求解 ----
    base_w = Pose(x=1.0, y=2.0, z=0.3,
                  qz=math.sin(math.radians(15)), qw=math.cos(math.radians(15)))
    target_w = ik.fk_world(q_true, base_w)
    ok3, q_w = ik.solve_world(target_w, base_w, [0.0] * ik.dof())
    res3 = ik.pose_error(world_to_base(target_w, base_w), q_w)
    print(f"world-frame IK: converged={ok3}, residual={res3:.3e} "
          f"-> {'[PASS]' if ok3 and res3 < 1e-3 else '[FAIL]'}")

    # ---- HMD Z 重锚定 (首帧 -> 1.2 m, 之后帧间增量) ----
    rb = GlobalPoseRebaser(z0=1.2)
    z1 = rb.update(Pose(z=0.37)).z
    z2 = rb.update(Pose(z=0.40)).z
    rb_pass = abs(z1 - 1.2) < 1e-9 and abs(z2 - 1.23) < 1e-9
    print(f"HMD rebase: first={z1:.3f} m, +0.03 -> {z2:.3f} m "
          f"-> {'[PASS]' if rb_pass else '[FAIL]'}")

    # ---- 全身 (腰+臂 13-DOF) IK + 躯干零空间任务 ----
    wb = WholeBodyIK(urdf, Arm.LEFT)
    q13 = list(_WAIST_BENT) + q_true
    wb_target = wb.fk(q13)
    ok4, q_wb = wb.solve(wb_target, [0.0] * wb.dof())
    res4 = wb.pose_error(wb_target, q_wb)
    print(f"whole-body IK (13-DOF): converged={ok4}, residual={res4:.3e} "
          f"-> {'[PASS]' if ok4 and res4 < 1e-3 else '[FAIL]'}")

    torso0 = wb.fk_torso(q13)
    wopt = WholeBodyOptions()
    wopt.use_torso_task = True
    wopt.torso_target = Pose(x=torso0.x, y=torso0.y, z=torso0.z - 0.03)
    ok5, q_t = wb.solve(wb_target, q13, wopt)
    res5 = wb.pose_error(wb_target, q_t)
    torso1 = wb.fk_torso(q_t)
    t_pass = ok5 and res5 < 1e-3 and \
        abs(torso1.z - wopt.torso_target.z) < 0.5 * 0.03
    print(f"torso task: chest z {torso0.z:.3f} -> {torso1.z:.3f} m "
          f"(target {wopt.torso_target.z:.3f}), pose_res={res5:.3e} "
          f"-> {'[PASS]' if t_pass else '[WARN]'}")

    all_ok = ok and res < 1e-3 and ok3 and res3 < 1e-3 \
        and rb_pass and ok4 and res4 < 1e-3
    return 0 if all_ok else 1


def main() -> int:
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
