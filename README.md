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

### 全局（世界）系目标

若目标位姿在全局系（如里程计 / 动捕 / HMD 世界系），传入 `base_link`
在全局系下的位姿，用 `*_world` 接口自动完成变换：

```python
from mozrobot_ik import Pose, world_to_base, base_to_world

base_w = Pose(x=1.0, y=2.0, z=0.3, qz=0.2588, qw=0.9659)  # base_link 在全局系
pose_w = ik.fk_world(q_current, base_w)                    # FK -> 全局系
ok, q_out = ik.solve_world(target_world, base_w, q_seed)   # 全局系目标求逆
```

### 弯腰时的单臂求解

手臂挂在腰腿末端；腰不在中性位时，先更新锁定配置再求解：

```python
ik.set_waist_q(q_waist6)        # 仅更新 6 个腰部关节 (其余保持中性)
ik.set_locked_q(q_full)         # 或传全身模型完整配置
ok, q_out = ik.solve(target, q_seed)
```

### HMD 追踪 Z 重锚定（Meta Quest 遥操作）

头显输出的地面平面不准，绝对 Z 不可信。`GlobalPoseRebaser` 把**首帧 Z
重置为躯干标称高度 1.2 m**，之后逐帧累加 Z 增量得到实时躯干高度：

```python
from mozrobot_ik import GlobalPoseRebaser

rb = GlobalPoseRebaser(z0=1.2, max_dz_per_frame=0.1)  # 可选单帧限幅抗跳变
pose = rb.update(hmd_pose)      # 每帧调用; 首帧 z -> 1.2, 之后按增量
rb.reset()                      # 操作员重新对中时重新锚定
```

### 全身 IK（腰 6-DOF + 单臂 7-DOF = 13-DOF）

`WholeBodyIK` 让末端借助腰部扩大工作空间。加权 DLS 默认让腰动得比臂少；
可选躯干零空间任务，让胸部（`LegWaist_Tip`）在不干扰末端的前提下跟踪目标
高度（如 HMD 重锚定输出）：

```python
from mozrobot_ik import WholeBodyIK, WholeBodyOptions

wb = WholeBodyIK(get_spirit01_urdf(), Arm.LEFT)   # q = [腰0..5, 臂0..6]
ok, q13 = wb.solve(target_pose, q_seed13)
q_waist, q_arm = wb.split_q(q13)                  # 拆分下发腰/臂指令

opt = WholeBodyOptions()
opt.use_torso_task = True
opt.torso_target = rb.update(hmd_pose)            # 胸部跟踪 HMD 高度
ok, q13 = wb.solve(target_pose, q_seed13, opt)
```

> 双臂遥操作注意：不要同时运行左右两个 `WholeBodyIK`（会争抢腰部）——
> 先解算/确定腰部，再用 `ArmIK.set_waist_q()` 分别解两臂。

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
pytest   # 离线单元测试: FK/IK 往返、psi 冗余、左右臂、全局系、重锚定、全身 IK
```

## 项目结构

```
mozrobot-ik/
├── mozrobot_ik/
│   ├── __init__.py        # 导出 ArmIK / WholeBodyIK / GlobalPoseRebaser / ...
│   ├── solver.py          # IK 核心 (DLS + psi 零空间 + 全身加权 DLS)
│   ├── frames.py          # 全局系变换 + HMD Z 重锚定
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
