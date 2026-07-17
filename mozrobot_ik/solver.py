#!/usr/bin/env python3
"""
本地笛卡尔→关节逆运动学 (Pinocchio Python)

在本地用 Pinocchio + URDF 计算逆解 (Cartesian -> Joint)，不依赖机器人控制器端 IK。
算法: 阻尼最小二乘 (DLS / Levenberg-Marquardt) + 7-DOF 臂角 psi 零空间控制。

坐标系: 解出的位姿相对 Pinocchio 模型根 base_link。请先用 fk() 校验
fk(当前关节) 是否复现实测 end_pose (详见 README 的"参考坐标系"一节)。
若目标在全局(世界)系, 用 fk_world()/solve_world() 并传入 base_link
在全局系下的位姿。

典型用法:
    from mozrobot_ik import ArmIK, IKOptions, Arm, get_spirit01_urdf

    ik = ArmIK(get_spirit01_urdf(), Arm.LEFT)
    ok, q_out = ik.solve(target_pose, q_seed)   # q_seed 用当前实测关节角

Copyright (c) 2025 Spirit AI. All rights reserved.
"""

import math

import numpy as np
import pinocchio

from .frames import pose_to_se3, se3_to_pose, world_to_base, base_to_world
from .types import Arm, Pose

_WAIST_JOINTS = [f"LegWaist-{i}" for i in range(6)]


class IKOptions:
    """IK 求解参数"""

    def __init__(self):
        self.max_iters = 1000     # 最大迭代次数
        self.eps = 1e-4           # 收敛阈值 (位姿误差范数)
        self.dt = 0.1             # 积分步长
        self.damp = 1e-6          # DLS 阻尼系数 λ²
        self.clamp = True         # 每步裁剪到关节限位
        # ---- 臂角 psi 冗余控制 (7-DOF 零空间, 仅 ArmIK) ----
        self.use_psi = False      # 启用臂角零空间控制
        self.psi_target = 0.0     # 目标臂角 (rad)
        self.psi_gain = 1.0       # 臂角收敛增益
        self.psi_eps = math.radians(0.5)  # 臂角收敛阈值 (rad)
        self.psi_ref = np.array([0.0, 0.0, 1.0])  # 参考向量(定义 psi=0 平面)


class WholeBodyOptions(IKOptions):
    """全身 (腰+臂 13-DOF) IK 求解参数"""

    def __init__(self):
        super().__init__()
        # 关节权重 (nv,): 权重越大该关节动得越少; None 用默认 (腰 4.0, 臂 1.0)
        self.joint_weights = None
        # ---- 躯干 (LegWaist_Tip) 位置零空间任务 ----
        self.use_torso_task = False   # 启用躯干次级任务
        self.torso_target = None      # 躯干目标位姿 (Pose, 取位置部分)
        self.torso_gain = 1.0         # 躯干任务增益
        self.torso_eps = 1e-3         # 躯干位置收敛阈值 (m)


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class _WorldFrameMixin:
    """全局(世界)系 API: 需要 base_link 在全局系下的位姿 (如里程计)。"""

    def fk_world(self, q, base_in_world: Pose) -> Pose:
        """正运动学 (全局系): 关节角 -> 末端在全局系下的位姿。"""
        return base_to_world(self.fk(q), base_in_world)

    def solve_world(self, target_world: Pose, base_in_world: Pose,
                    q_seed, opts: IKOptions = None):
        """逆运动学 (全局系): 先把全局目标变换到 base_link 系再求解。"""
        return self.solve(world_to_base(target_world, base_in_world),
                          q_seed, opts)


class ArmIK(_WorldFrameMixin):
    """单臂逆运动学求解器 (基于 Pinocchio 精简模型)"""

    def __init__(self, urdf_xml: str, arm: Arm, locked_q=None):
        """
        :param urdf_xml: URDF 字符串 (如 get_spirit01_urdf())
        :param arm:      目标手臂 (Arm.LEFT / Arm.RIGHT)
        :param locked_q: 其余关节(腰腿/底盘)在全身模型中的参考配置; 默认中性。
                         若 end_pose 相对 base_link 且腰腿非零, 应传入对应全身 q
                         (之后也可用 set_locked_q()/set_waist_q() 更新)。
        """
        # 1. 加载完整模型 (固定基座)
        self._full = pinocchio.buildModelFromXML(urdf_xml)

        # 2. 目标臂关节名 / 末端坐标系
        prefix = arm.value
        self._prefix = prefix
        self.ee_frame_name = prefix + "_ee"
        arm_joints = [f"{prefix}-{i}" for i in range(7)]

        # 3. 锁定除目标臂外的所有关节 -> 精简模型
        self._lock_ids = [jid for jid in range(1, len(self._full.names))
                          if self._full.names[jid] not in arm_joints]
        q_ref = locked_q if (locked_q is not None
                             and len(locked_q) == self._full.nq) \
            else pinocchio.neutral(self._full)
        self._rebuild(np.array(q_ref, dtype=float))

    def _rebuild(self, q_ref):
        """按锁定参考配置 q_ref 重建精简模型并刷新坐标系索引/限位。"""
        self.model = pinocchio.buildReducedModel(self._full, self._lock_ids,
                                                 q_ref)
        self.data = self.model.createData()

        prefix = self._prefix
        self.ee_id = self.model.getFrameId(self.ee_frame_name)
        # 臂角几何: 肩(link1) / 肘(link3) / 腕(link5) 三点确定臂平面
        self.shoulder_id = self.model.getFrameId(prefix + "_link1")
        self.elbow_id = self.model.getFrameId(prefix + "_link3")
        self.wrist_id = self.model.getFrameId(prefix + "_link5")

        self.lower = np.array(self.model.lowerPositionLimit)
        self.upper = np.array(self.model.upperPositionLimit)

    def set_locked_q(self, q_full):
        """更新锁定关节 (腰腿/底盘) 的配置并重建精简模型。

        :param q_full: 全身模型配置向量 (长度 = 全模型 nq)
        """
        q_ref = np.array(q_full, dtype=float)
        if len(q_ref) != self._full.nq:
            raise ValueError(
                f"q_full 长度应为 {self._full.nq}, 实际 {len(q_ref)}")
        self._rebuild(q_ref)

    def set_waist_q(self, q_waist):
        """仅更新腰部 (LegWaist-0..5) 锁定角度, 其余锁定关节保持中性。

        :param q_waist: 6 个腰部关节角 (rad)
        """
        if len(q_waist) != len(_WAIST_JOINTS):
            raise ValueError(f"q_waist 长度应为 {len(_WAIST_JOINTS)}, "
                             f"实际 {len(q_waist)}")
        q_ref = pinocchio.neutral(self._full)
        for name, angle in zip(_WAIST_JOINTS, q_waist):
            jid = self._full.getJointId(name)
            q_ref[self._full.joints[jid].idx_q] = float(angle)
        self._rebuild(q_ref)

    def dof(self) -> int:
        """自由度数 (= 7)"""
        return self.model.nq

    def fk(self, q) -> Pose:
        """正运动学: 关节角 -> 末端位姿 (相对 base_link)。下发前务必先校验参考系。"""
        qv = np.array(q, dtype=float)
        pinocchio.forwardKinematics(self.model, self.data, qv)
        pinocchio.updateFramePlacement(self.model, self.data, self.ee_id)
        return se3_to_pose(self.data.oMf[self.ee_id])

    def pose_error(self, target: Pose, q) -> float:
        """末端位姿误差范数 (位置+姿态 6D)"""
        qv = np.array(q, dtype=float)
        data = self.model.createData()
        pinocchio.forwardKinematics(self.model, data, qv)
        pinocchio.updateFramePlacement(self.model, data, self.ee_id)
        iMd = data.oMf[self.ee_id].actInv(pose_to_se3(target))
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

        oMdes = pose_to_se3(target)
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


class WholeBodyIK(_WorldFrameMixin):
    """全身逆运动学求解器: 腰 (LegWaist 6-DOF) + 单臂 (7-DOF) = 13-DOF。

    末端可借助腰部运动扩大工作空间。加权 DLS 默认让腰动得比臂少
    (joint_weights), 可选躯干位置零空间任务 (use_torso_task) 让胸部
    (LegWaist_Tip) 在不干扰末端的前提下跟踪目标高度 (如 HMD 重锚定输出)。

    注意: 双臂遥操作时不要同时运行左右两个 WholeBodyIK (会争抢腰部) —
    先解算/确定腰部, 再用 ArmIK.set_waist_q() 分别解两臂。

    关节顺序: q = [LegWaist-0..5, {arm}-0..6], 用 split_q() 拆分下发。
    """

    TORSO_FRAME = "LegWaist_Tip"

    def __init__(self, urdf_xml: str, arm: Arm, locked_q=None):
        """
        :param urdf_xml: URDF 字符串 (如 get_spirit01_urdf())
        :param arm:      目标手臂 (Arm.LEFT / Arm.RIGHT)
        :param locked_q: 其余关节(另一臂/底盘)在全身模型中的参考配置; 默认中性。
        """
        full = pinocchio.buildModelFromXML(urdf_xml)

        prefix = arm.value
        self.ee_frame_name = prefix + "_ee"
        keep = set(_WAIST_JOINTS + [f"{prefix}-{i}" for i in range(7)])

        q_ref = locked_q if (locked_q is not None and len(locked_q) == full.nq) \
            else pinocchio.neutral(full)
        lock_ids = [jid for jid in range(1, len(full.names))
                    if full.names[jid] not in keep]
        self.model = pinocchio.buildReducedModel(full, lock_ids, q_ref)
        self.data = self.model.createData()

        self.ee_id = self.model.getFrameId(self.ee_frame_name)
        self.torso_id = self.model.getFrameId(self.TORSO_FRAME)

        # 腰部关节在精简模型中的速度索引 (用于默认权重)
        self._waist_idx_v = [self.model.joints[self.model.getJointId(n)].idx_v
                             for n in _WAIST_JOINTS]
        self.torso_dof = len(self._waist_idx_v)

        self.lower = np.array(self.model.lowerPositionLimit)
        self.upper = np.array(self.model.upperPositionLimit)

    def dof(self) -> int:
        """自由度数 (= 13)"""
        return self.model.nq

    def split_q(self, q):
        """q (13,) -> (q_waist[6], q_arm[7]), 便于分别下发腰/臂指令。"""
        q = [float(x) for x in q]
        return q[:self.torso_dof], q[self.torso_dof:]

    def fk(self, q) -> Pose:
        """正运动学: 关节角 -> 末端位姿 (相对 base_link)。"""
        qv = np.array(q, dtype=float)
        pinocchio.forwardKinematics(self.model, self.data, qv)
        pinocchio.updateFramePlacement(self.model, self.data, self.ee_id)
        return se3_to_pose(self.data.oMf[self.ee_id])

    def fk_torso(self, q) -> Pose:
        """正运动学: 关节角 -> 躯干 (LegWaist_Tip) 位姿 (相对 base_link)。"""
        qv = np.array(q, dtype=float)
        pinocchio.forwardKinematics(self.model, self.data, qv)
        pinocchio.updateFramePlacement(self.model, self.data, self.torso_id)
        return se3_to_pose(self.data.oMf[self.torso_id])

    def pose_error(self, target: Pose, q) -> float:
        """末端位姿误差范数 (位置+姿态 6D)"""
        qv = np.array(q, dtype=float)
        data = self.model.createData()
        pinocchio.forwardKinematics(self.model, data, qv)
        pinocchio.updateFramePlacement(self.model, data, self.ee_id)
        iMd = data.oMf[self.ee_id].actInv(pose_to_se3(target))
        return float(np.linalg.norm(pinocchio.log6(iMd).vector))

    def default_weights(self):
        """默认关节权重: 腰 4.0 / 臂 1.0 (权重大 -> 动得少)。"""
        w = np.ones(self.model.nv)
        for i in self._waist_idx_v:
            w[i] = 4.0
        return w

    def solve(self, target: Pose, q_seed, opts: WholeBodyOptions = None):
        """
        逆运动学: 末端目标位姿 -> 13 关节角 (加权 DLS + 可选躯干零空间任务)
        :return: (converged: bool, q_out: list[float])
        """
        if opts is None:
            opts = WholeBodyOptions()

        weights = np.array(opts.joint_weights, dtype=float) \
            if getattr(opts, "joint_weights", None) is not None \
            else self.default_weights()
        Winv = np.diag(1.0 / weights)

        use_torso = bool(getattr(opts, "use_torso_task", False)) \
            and getattr(opts, "torso_target", None) is not None
        if use_torso:
            p_torso_des = np.array([opts.torso_target.x,
                                    opts.torso_target.y,
                                    opts.torso_target.z])

        oMdes = pose_to_se3(target)
        if q_seed is not None and len(q_seed) == self.model.nq:
            q = np.array(q_seed, dtype=float)
        else:
            q = pinocchio.neutral(self.model)

        data = self.model.createData()
        I6 = np.eye(6)
        I3 = np.eye(3)
        I_nv = np.eye(self.model.nv)
        success = False
        prev_torso_err = None

        for _ in range(opts.max_iters):
            pinocchio.forwardKinematics(self.model, data, q)
            pinocchio.updateFramePlacements(self.model, data)

            # 主任务误差 (末端局部系)
            iMd = data.oMf[self.ee_id].actInv(oMdes)
            err = pinocchio.log6(iMd).vector
            pose_ok = np.linalg.norm(err) < opts.eps

            # 躯干位置误差 (次级任务, 尽力而为)
            torso_done = True
            if use_torso:
                p_torso = np.array(data.oMf[self.torso_id].translation)
                e_torso = p_torso_des - p_torso
                torso_err = float(np.linalg.norm(e_torso))
                # 收敛 或 零空间内已无法再改善 (误差停滞) 即认为完成
                torso_done = torso_err < opts.torso_eps or (
                    prev_torso_err is not None
                    and abs(prev_torso_err - torso_err) < 1e-7)
                prev_torso_err = torso_err

            if pose_ok and torso_done:
                success = True
                break

            # 主任务雅可比 + Jlog6 修正
            J = pinocchio.computeFrameJacobian(
                self.model, data, q, self.ee_id, pinocchio.ReferenceFrame.LOCAL)
            J = -pinocchio.Jlog6(iMd.inverse()) @ J

            # 加权 DLS: v = -W⁻¹Jᵀ (J W⁻¹ Jᵀ + λI)⁻¹ err
            JWJt = J @ Winv @ J.T + opts.damp * I6
            Jpinv = Winv @ J.T @ np.linalg.solve(JWJt, I6)
            v = -Jpinv @ err

            if use_torso and not torso_done:
                # 躯干位置任务投影到主任务零空间 (不干扰末端位姿)
                Jt = pinocchio.computeFrameJacobian(
                    self.model, data, q, self.torso_id,
                    pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
                vt = Jt.T @ np.linalg.solve(Jt @ Jt.T + opts.damp * I3,
                                            e_torso)
                N = I_nv - Jpinv @ J
                v = v + N @ (opts.torso_gain * vt)

            q = pinocchio.integrate(self.model, q, v * opts.dt)

            if opts.clamp:
                q = np.minimum(np.maximum(q, self.lower), self.upper)

        return success, [float(x) for x in q]
