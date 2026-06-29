#!/usr/bin/env python3
"""
命令行入口
Copyright (c) 2025 Spirit AI. All rights reserved.
"""

import math
import sys

from .solver import ArmIK, IKOptions
from .types import Arm
from .urdf_data import get_spirit01_urdf

_LEFT_ARM_WORK_POINT_DEG = [-9, -50, -20, -90, -35, 8, -7]


def selftest() -> int:
    """离线自检: FK/IK 往返 + 臂角 psi 控制 (无需机器人)。"""
    print("=== mozrobot-ik self-test (offline) ===")
    ik = ArmIK(get_spirit01_urdf(), Arm.LEFT)
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

    return 0 if (ok and res < 1e-3) else 1


def main() -> int:
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
