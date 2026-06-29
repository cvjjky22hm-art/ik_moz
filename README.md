# mozrobot-ik

Spirit01 双臂机器人的**本地笛卡尔→关节逆运动学**（Inverse Kinematics）。

在本地用 [Pinocchio](https://github.com/stack-of-tasks/pinocchio) + URDF 计算逆解
（Cartesian → Joint），**不依赖机器人控制器端 IK**。给定末端目标位姿，求出 7 个手臂关节角，
可用于离线规划、可达性/限位预检、以及把结果以关节指令下发给机器人。

- 核心库**仅依赖 `pinocchio` + `numpy`**，与机器人 SDK（mozrobot）解耦，离线即可运行。
- 算法：阻尼最小二乘（DLS / Levenberg–Marquardt）+ 7-DOF 冗余臂角 `psi` 零空间控制。
- 支持左/右臂，自动应用 URDF 关节限位。

## 安装

```bash
# 依赖 (Pinocchio 的 PyPI 包名为 pin)
pip install -r requirements.txt
# 或安装本包
pip install -e .
```

> Pinocchio 在 Linux / macOS(arm64) 均有预编译 wheel，`pip install pin` 即可。

## 快速开始

```bash
# 离线自检 (无需机器人)
python examples/selftest.py
# 或安装后:
mozrobot-ik-selftest
```

预期输出（已用 Pinocchio 3.9 + spirit01.urdf 实测）：

```
=== mozrobot-ik self-test (offline) ===
Arm DOF=7, ee_frame=LeftArm_ee
IK converged=True, residual=9.168e-05
[PASS] pose IK
psi: start=-113.1 target=-93.1 got=-93.6 deg | pose_res=6.838e-07
[PASS] psi control
```

## 使用 API

```python
from mozrobot_ik import ArmIK, IKOptions, Arm, get_spirit01_urdf

ik = ArmIK(get_spirit01_urdf(), Arm.LEFT)

# 正解: 关节角 -> 位姿 (下发前先用它校验参考系)
pose = ik.fk(q_current)

# 逆解: 目标位姿 -> 关节角
ok, q_out = ik.solve(target_pose, q_seed=q_current)

# 臂角 psi 冗余控制: 保持末端位姿不变, 把肘部臂角驱动到目标
import math
opt = IKOptions()
opt.use_psi = True
opt.psi_target = math.radians(30.0)
ok, q_out = ik.solve(target_pose, q_seed, opt)
psi = ik.arm_angle(q_out)
```

`Pose` 字段：`x, y, z`（米）+ `qx, qy, qz, qw`（单位四元数）。

## 算法

```
给定 目标位姿 oMdes、种子关节角 q0
重复:
  1. forwardKinematics(q) + updateFramePlacement(ee)
  2. err = log6( oMcur⁻¹ · oMdes )                       # 6维位姿误差(末端局部系)
  3. 收敛判据: ||err|| < eps  (启用 psi 时还需 |Δpsi| < psi_eps)
  4. J = computeFrameJacobian(q, ee, LOCAL);  J ← -Jlog6(err)·J
  5. DLS:  v = -Jᵀ (J Jᵀ + λ²I)⁻¹ err
     (psi) v += (I - J⁺J) · gain·(psi_target - psi)·(dψ/dq)   # 零空间, 不影响末端
  6. q ← integrate(q, v·dt);  clamp 到关节限位
```

**臂角 psi** 定义：由肩(link1)/肘(link3)/腕(link5)三点构成的臂平面，与参考平面
（`psi_ref` 向量，默认世界 +Z，与肩-腕轴张成）绕肩-腕轴的有符号夹角。`dψ/dq` 用前向差分数值求得。

## 参考坐标系（重要）

IK 解出的末端位姿相对 **Pinocchio 模型根 `base_link`**，且手臂挂在腰腿末端，
末端位姿取决于腰腿当前角度。务必保证「目标 `end_pose`」与 FK 参考系一致：

- 先用 `ik.fk(当前关节角)` 对比机器人回报的 `end_pose`，残差应很小；
- 若腰腿非零且 `end_pose` 相对 `base_link`，构造 `ArmIK` 时传入 `locked_q`（全身参考配置）。

## 与 mozrobot SDK 集成

`examples/robot_integration.py` 展示完整链路（读状态 → 本地 IK → `use_jnt=True` 下发）。
核心库不导入 mozrobot；集成示例按需 `import mozrobot`，需已安装该绑定。

## 测试

```bash
pip install pytest
pytest          # 离线单元测试: FK/IK 往返、psi 冗余、左右臂
```

## 项目结构

```
mozrobot-ik/
├── mozrobot_ik/
│   ├── __init__.py        # 导出 ArmIK / IKOptions / Arm / Pose / get_spirit01_urdf
│   ├── solver.py          # IK 核心 (DLS + psi 零空间)
│   ├── types.py           # Pose / Arm (无 SDK 依赖)
│   ├── urdf_data.py       # URDF 加载
│   ├── cli.py             # 命令行自检入口
│   └── data/spirit01.urdf # 机器人模型
├── examples/
│   ├── selftest.py        # 离线自检
│   └── robot_integration.py  # 与 mozrobot SDK 集成
├── tests/test_ik.py       # pytest
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 后续可扩展

- 解析 IK（S-R-S 7-DOF 闭式解，臂角作显式输入），精度/速度更优。
- 实时跟踪：把 `dψ/dq` 改为解析式以降低每帧 FK 开销。
- 双臂批量求解、可达性/碰撞预检。

---
Copyright (c) 2025 Spirit AI. All rights reserved.
