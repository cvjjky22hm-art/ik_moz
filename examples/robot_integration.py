#!/usr/bin/env python3
"""
与 mozrobot SDK 集成示例 (需要已安装 mozrobot 绑定)

展示完整链路: 读机器人当前状态 -> 本地 IK 解算 -> use_jnt=True 下发。
本文件按需导入 mozrobot, 不影响核心库的独立性。

用法:
    python examples/robot_integration.py <interface_ip> [--send]
Copyright (c) 2025 Spirit AI. All rights reserved.
"""

import sys
import time

from mozrobot_ik import ArmIK, Arm, Pose, get_spirit01_urdf

try:
    import mozrobot
except ImportError:
    print("需要 mozrobot SDK 绑定; 离线请改用 examples/selftest.py")
    sys.exit(1)


def sdk_pose_to_pose(sp) -> Pose:
    """mozrobot.Pose -> mozrobot_ik.Pose"""
    return Pose(x=sp.x, y=sp.y, z=sp.z, qx=sp.qx, qy=sp.qy, qz=sp.qz, qw=sp.qw)


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <interface_ip> [--send]")
        return 1
    interface_ip = sys.argv[1]
    send = "--send" in sys.argv[2:]

    # ---- 连接 (具体 API 以 mozrobot SDK 为准) ----
    net = mozrobot.Network.get_instance()
    net.init(interface_ip)
    # 这里省略等待连接的细节, 详见 SDK examples_python

    # 1. 读左臂当前状态
    states = mozrobot.States()
    ret, mu_states = states.get_mech_unit_state()
    if ret != 0:
        print("读取状态失败")
        return 1
    left = next((s for s in mu_states
                 if s.type == mozrobot.MechUnitType.LEFT_ARM), None)
    if left is None or len(left.jnt_pos) == 0:
        print("左臂状态不可用")
        return 1

    ik = ArmIK(get_spirit01_urdf(), Arm.LEFT)
    q_seed = list(left.jnt_pos)

    # 2. 参考系自检
    frame_err = ik.pose_error(sdk_pose_to_pose(left.end_pose), q_seed)
    print(f"frame check residual={frame_err:.3e} "
          f"{'(OK)' if frame_err < 1e-2 else '(WARN: 参考系不一致)'}")

    # 3. 目标: 当前位姿沿 X +5cm
    target = sdk_pose_to_pose(left.end_pose)
    target.x += 0.05

    # 4. 求解
    ok, q_out = ik.solve(target, q_seed)
    print(f"IK converged={ok}, residual={ik.pose_error(target, q_out):.3e}")
    print(f"q_ik = {q_out}")
    if not ok:
        print("IK 未收敛 (目标不可达?), 中止")
        return 1

    # 5. 下发 (use_jnt=True)
    if send:
        robot = mozrobot.Robot()
        robot.switch_task_mode(mozrobot.TaskMode.NORMAL)
        motion = mozrobot.Motion()
        cmd = mozrobot.MechUnitPosCmd()
        cmd.type = mozrobot.MechUnitType.LEFT_ARM
        cmd.jnt_pos = q_out
        cmd.use_jnt = True
        if motion.start_move([cmd]) != 0:
            print("下发失败")
            return 1
        while not motion.is_idle():
            time.sleep(1)
        print("Move done")
    else:
        print("Dry-run (加 --send 才真正下发)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
