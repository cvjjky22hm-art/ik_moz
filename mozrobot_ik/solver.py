#!/usr/bin/env python3
"""
本地笛卡尔→关节逆运动学 (Pinocchio Python)

在本地用 Pinocchio + URDF 计算逆解 (Cartesian -> Joint)，不依赖机器人控制器端 IK。
算法: 阻尼最小二乘 (DLS / Levenberg-Marquardt) + 7-DOF 臂角 psi 零空间控制。

坐标系: 解出的位姿相对 Pinocchio 模型根 base_link。请先用 fk() 校验
fk(当前关节) 是否复现实测 end_pose (详见 README 的"参考坐标系"一节)。

典型用法:
    from mozrobot_ik import ArmIK, IKOptions, Arm, get_spirit01_urdf

    ik = ArmIK(get_spirit01_urdf(), Arm.LEFT)
    ok, q_out = ik.solve(target_pose, q_seed)   # q_seed 用当前实测关节角

Copyright (c) 2025 Spirit AI. All rights reserved.
"""

import math

import numpy as np
import pinocchio

from .types import Arm, Pose


class IKOptions:
    """IK 求解参数"""

    def __init__(self):
        self.max_iters = 1000     # 最大迭代次数
        self.eps = 1e-4           # 收敛阈值 (位姿误差范数)
        self.dt = 0.1             # 积分步长
        self.damp = 1e-6          # DLS 阻尼系数 λ²
        self.clamp = True         # 每步裁剪到关节限位
        # ---- 臂角 psi 冗余控制 (7-DOF 零空间) ----
        self.use_psi = False      # 启用臂角零空间控制
        self.psi_target = 0.0     # 目标臂角 (rad)
        self.psi_gain = 1.0       # 臂角收敛增益
        self.psi_eps = math.radians(0.5)  # 臂角收敛阈值 (rad)
        self.psi_ref = np.array([0.0, 0.0, 1.0])  # 参考向量(定义 psi=0 平面)


def _pose_to_se3(p: Pose):
    """Pose -> pinocchio.SE3"""
    quat = pinocchio.Quaternion(p.qw, p.qx, p.qy, p.qz)
    quat.normalize()
    return pinocchio.SE3(quat.matrix(), np.array([p.x, p.y, p.z]))


def _se3_to_pose(T) -> Pose:
    """pinocchio.SE3 -> Pose"""
    t = T.translation
    quat = pinocchio.Quaternion(T.rotation)
    quat.normalize()
    return Pose(
        x=float(t[0]), y=float(t[1]), z=float(t[2]),
        qx=float(quat.x), qy=float(quat.y), qz=float(quat.z), qw=float(quat.w),
    )


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class ArmIK:
    """单臂逆运动学求解器 (基于 Pinocchio 精简模型)"""

    def __init__(self, urdf_xml: str, arm: Arm, locked_q=None):
        """
        :param urdf_xml: URDF 字符串 (如 get_spirit01_urdf())
        :param arm:      目标手臂 (Arm.LEFT / Arm.RIGHT)
        :param locked_q: 其余关节(腰腿/底盘)在全身模型中的参考配置; 默认中性。
                         若 end_pose 相对 base_link 且腰腿非零, 应传入对应全身 q。
        """
        # 1. 加载完整模型 (固定基座)
        full = pinocchio.buildModelFromXML(urdf_xml)

        # 2. 目标臂关节名 / 末端坐标系
        prefix = arm.value
        self.ee_frame_name = prefix + "_ee"
        arm_joints = [f"{prefix}-{i}" for i in range(7)]

        # 3. 锁定除目标臂外的所有关节 -> 精简模型
        q_ref = locked_q if (locked_q is not None and len(locked_q) == full.nq) \
            else pinocchio.neutral(full)
        lock_ids = [jid for jid in range(1, len(full.names))
                    if full.names[jid] not in arm_joints]
        self.model = pinocchio.buildReducedModel(full, lock_ids, q_ref)
        self.data = self.model.createData()

        self.ee_id = self.model.getFrameId(self.ee_frame_name)
        # 臂角几何: 肩(link1) / 肘(link3) / 腕(link5) 三点确定臂平面
        self.shoulder_id = self.model.getFrameId(prefix + "_link1")
        self.elbow_id = self.model.getFrameId(prefix + "_link3")
        self.wrist_id = self.model.getFrameId(prefix + "_link5")

        self.lower = np.array(self.model.lowerPositionLimit)
        self.upper = np.array(self.model.upperPositionLimit)

    def dof(self) -> int:
        """自由度数 (= 7)"""
        return self.model.nq

    def fk(self, q) -> Pose:
        """正运动学: 关节角 -> 末端位姿 (相对 base_link)。下发前务必先校验参考系。"""
        qv = np.array(q, dtype=float)
        pinocchio.forwardKinematics(self.model, self.data, qv)
        pinocchio.updateFramePlacement(self.model, self.data, self.ee_id)
        return _se3_to_pose(self.data.oMf[self.ee_id])

    def pose_error(self, target: Pose, q) -> float:
        """末端位姿误差范数 (位置+姿态 6D)"""
        qv = np.array(q, dtype=float)
        data = self.model.createData()
        pinocchio.forwardKinematics(self.model, data, qv)
        pinocchio.updateFramePlacement(self.model, data, self.ee_id)
        iMd = data.oMf[self.ee_id].actInv(_pose_to_se3(target))
        return float(np.linalg.norm(pinocchio.log6(iMd).vector))

    def arm_angle(self, q, psi_ref=None) -> float:
        """当前臂角 psi (rad)"""
        if psi_ref is None:
            psi_ref = np.array([0.0, 0.0, 1.0])
        return self._arm_angle(np.array(q, dtype=float), psi_ref)

    def solve(self, target: Pose, q_seed, opts: IKOptions = None):
        """
        逆运动学: 目标位姿 -> 关节角 (阻尼最小二乘 + 可选臂角零空间)
        :return: (converged: bool, q_out: list[float])
        """
        if opts is None:
            opts = IKOptions()

        oMdes = _pose_to_se3(target)
        if q_seed is not None and len(q_seed) == self.model.nq:
            q = np.array(q_seed, dtype=float)
        else:
            q = pinocchio.neutral(self.model)

        data = self.model.createData()
        I6 = np.eye(6)
        I_nv = np.eye(self.model.nv)
        success = False

        for _ in range(opts.max_iters):
            pinocchio.forwardKinematics(self.model, data, q)
            pinocchio.updateFramePlacement(self.model, data, self.ee_id)

            # 误差 (末端局部系): err = log6( oMcur⁻¹ · oMdes )
            iMd = data.oMf[self.ee_id].actInv(oMdes)
            err = pinocchio.log6(iMd).vector
            pose_ok = np.linalg.norm(err) < opts.eps

            # 臂角误差 (零空间任务)
            dpsi = 0.0
            psi_ok = True
            if opts.use_psi:
                psi_cur = self._arm_angle(q, opts.psi_ref)
                dpsi = _wrap_pi(opts.psi_target - psi_cur)
                psi_ok = abs(dpsi) < opts.psi_eps

            # 末端位姿与臂角都满足才收敛 (否则即使位姿已到, 仍需继续调臂角)
            if pose_ok and psi_ok:
                success = True
                break

            # 雅可比 + Jlog6 修正
            J = pinocchio.computeFrameJacobian(
                self.model, data, q, self.ee_id, pinocchio.ReferenceFrame.LOCAL)
            J = -pinocchio.Jlog6(iMd.inverse()) @ J

            # DLS 阻尼最小二乘
            JJt = J @ J.T + opts.damp * I6

            if opts.use_psi:
                # 阻尼伪逆 Jpinv = Jᵀ (JJt)⁻¹  (nv x 6)
                Jpinv = J.T @ np.linalg.solve(JJt, I6)
                v = -Jpinv @ err
                # 零空间投影 N = I - Jpinv·J: 推臂角到 psi_target 而不影响末端位姿
                N = I_nv - Jpinv @ J
                grad = self._psi_gradient(q, psi_cur, opts.psi_ref)
                v = v + N @ (opts.psi_gain * dpsi * grad)
            else:
                v = -J.T @ np.linalg.solve(JJt, err)

            q = pinocchio.integrate(self.model, q, v * opts.dt)

            if opts.clamp:
                q = np.minimum(np.maximum(q, self.lower), self.upper)

        return success, [float(x) for x in q]

    # ---------------- 内部实现 ----------------

    def _arm_angle(self, q, psi_ref):
        data = self.model.createData()
        pinocchio.forwardKinematics(self.model, data, q)
        pinocchio.updateFramePlacements(self.model, data)
        S = np.array(data.oMf[self.shoulder_id].translation)
        E = np.array(data.oMf[self.elbow_id].translation)
        W = np.array(data.oMf[self.wrist_id].translation)
        xsw = W - S
        n = np.linalg.norm(xsw)
        if n < 1e-9:
            return 0.0
        xsw = xsw / n
        e_perp = (E - S) - np.dot(E - S, xsw) * xsw
        ref = psi_ref - np.dot(psi_ref, xsw) * xsw
        if np.linalg.norm(e_perp) < 1e-9 or np.linalg.norm(ref) < 1e-9:
            return 0.0
        y = np.dot(np.cross(ref, e_perp), xsw)  # 绕 xsw 的有符号分量
        x = np.dot(ref, e_perp)
        return math.atan2(y, x)

    def _psi_gradient(self, q, psi0, psi_ref):
        """臂角对关节角的数值梯度 dψ/dq (前向差分)"""
        h = 1e-6
        g = np.zeros(self.model.nv)
        for i in range(self.model.nv):
            qp = q.copy()
            qp[i] += h
            g[i] = _wrap_pi(self._arm_angle(qp, psi_ref) - psi0) / h
        return g
