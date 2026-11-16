# 双主机复现步骤

这是2026-11-12两次成功试验的原部署顺序。以下命令仅供现场准备完成后使用，仓库检查和CI不会执行它们。

## 1. 配置基线

| 项目 | 成功值 |
|---|---|
| 模型 | run02 EMA，第30,000步 |
| 时间融合 | `temporal_ensemble_coeff=0.01` |
| 动作块/每步感知 | `chunk_size=100`、`n_action_steps=1` |
| 策略频率 | 30Hz |
| 夹爪状态屏蔽 | `mask_gripper_state=true` |
| 训练动作提前量 | 10帧，运行时无额外偏移 |
| 头部相机 | `243722073540`，640×480 RGB，center预处理 |
| 腕部相机 | `135122078001`，1280×720 RGB，pad预处理 |
| 模型图像大小 | 64×64 |
| 夹爪动作 | 阈值0.5、滞回0.05 |
| 宽度标签 | 闭合阈值0.0693m、打开阈值0.074m |
| 程序每轴速度上限 | 线速度0.12m/s、角速度0.35rad/s |
| 自动回位/夹爪重置 | 关闭/保持当前状态 |
| ROS域 | Yin为66，lixin为0 |

实际链路：双RGB和真实关节状态 → ACT动作块 → 原生时间融合 → 7关节目标经FK转末端目标 → 有界增量 → UDP8888 `<7f>` → 原笛卡尔控制器及夹爪动作服务。FRST状态通过UDP8896返回。

原模型输出不是末端增量，不能把7个关节目标直接作为UDP增量发送。

## 2. 已验证环境与路径

- lixin：Ubuntu22.04、ROS Humble、Python3.10.19，原`tools`环境；PyTorch2.5.1+cu124、torchvision0.20.1+cu124。
- Yin：实时内核`5.15.0-1098-realtime`、ROS Humble、libfranka0.20.4及已构建ROS2工作区。
- 模型及原启动目录：`/home/lixin/act_gripper_temporal_run02_20261112`。
- Yin控制参数目录：`/home/yin/act_run02_baseline_control_20261112`。
- Python包版本完整记录见`environment/verified_versions.json`。Pinocchio发行包名是`pin`3.8.0。

本仓库的原运行代码保留这些路径及原完整性检查。新机器需要核对系统依赖、资产和路径，并重新进行离线验证；未提供经过新机器实测的一键安装器。

## 3. 相机与状态接收

在lixin单独终端启动已校准接收器：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
cd /data/TS-WAM
/data/TS-WAM/.venv/bin/python -m deployment.tswam_state_relay_receiver \
  --bind-host 0.0.0.0 --port 8896 --source-host 10.10.10.2 \
  --gripper-closed-width-threshold 0.0693
```

在另一终端启动相机：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
bash /data/Yin_ros2_ws/src/camera/start_cameras.sh --color --head --wrist --display
```

使用`/camera/cam_head/color/image_raw`和`/camera/cam_wrist/color/image_raw`。保留腕部resize默认关闭及上述分辨率。

## 4. 原笛卡尔控制器

在Yin上确认只有一套外部控制、无SpaceMouse动作发送后：

```bash
export ROS_DOMAIN_ID=66
export FRANKA_GRIPPER_CLOSED_WIDTH_THRESHOLD=0.0693
bash /home/yin/act_run02_baseline_control_20261112/run_control.sh cartesian load_gripper:=true
```

此脚本使用`controllers_act.yaml`：线速度和角速度死区为0，其他原参数保持。原SpaceMouse安全配置与这个ACT副本需区分。

## 5. 起点与检查

成功轮起点取自第一轮日志第0帧：

```text
q(rad) = [0.619065, 0.554681, -0.401264, -1.533722,
          0.273446, 2.165253, 0.913639]
q(deg) = [35.4698, 31.7809, -22.9907, -87.8758,
          15.6673, 124.0599, 52.3477]
```

方块位于示教起点，夹爪张开。先检查七关节、状态/图像持续更新、模型首预测。两次成功前首目标误差约4.0mm和5.2mm，是本场景记录，不是通用安全阈值。

在已部署的lixin目录检查：

```bash
bash /home/lixin/act_gripper_temporal_run02_20261112/run_real.sh check
```

完整记录观测回放还需要原数据和缓存：

```bash
bash /home/lixin/act_gripper_temporal_run02_20261112/offline_check.sh
```

原完整性清单会检查EMA、输入夹具、Parquet及原项目文件。`tools/verify_repository.py`检查的是Git代码快照；这两类检查范围不同。

## 6. 实机程序等待现场B

在Linux桌面、具有DISPLAY及真实TTY的终端：

```bash
bash /home/lixin/act_gripper_temporal_run02_20261112/run_real.sh run
```

等待`Press B to start; C exits.`。B启动/结束一轮，C退出，无自动结束时限。等待B前原程序探测桥接器并发送零机械臂增量、保持夹爪标签。自动回位关闭，默认reset任务名称不参与本次抓放。

结束后先让ACT退出并确认日志，再结束旧外部控制流程，随后切换SpaceMouse或手动引导。相机、状态接收器在各自终端Ctrl+C结束。保持单一动作发送端。

## 7. 七关节回位源码

`yin_controller/return_to_success_start/return_success_start.cpp`是RETEST 2前实际验证的单次工具，调用原版libfranka0.20.4 `MotionGenerator`，速度因子0.005。当次规划7.375秒、末端路径约5.2mm、实际最大关节速度约0.016rad/s。

它绑定固定的**预期起点**并检查静止、夹爪张开、位置包络与速度；不是通用回位器。当前姿态不同时，不能删除检查强行执行。

`--plan <csv>`只计算轨迹；`--execute <csv>`会真实运动。编译依赖见对应源码和上游运动生成器。本仓库CI不会编译或执行硬件动作。

## 8. 训练参考与复验边界

训练入口：`training_reference/scripts/block_pick_place/train_block_pick_place_v3_run02.sh`。保留原路径和新输出目录保护，调用`pretrain_act_bc.py`及原run02配置。

需要重训时，在原环境使用新输出目录；不覆盖现有成功检查点。源episode6的排除映射见`runtime_lixin/recorded_metadata/episode_selection.json`，新编号6对应源编号7。

两轮闭合时刻仅差0.056秒，松开时刻差10.535秒，释放末端位置差13.195mm、最终位置差22.075mm。保留该版本作对照，后续应记录完整任务的释放位置与时机分布。

`runtime_lixin/ACT_CONTROL_PROFILE_20261112.txt`保留了早期baseline说明；其中旧baseline启动路径不是本次成功入口，以本文件的temporal目录为准。
