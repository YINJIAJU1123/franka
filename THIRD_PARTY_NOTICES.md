# 第三方源码与许可

本仓库保留原源码及许可文本，没有通过根目录的统一许可证重新授权所有来源代码。

| 来源 | 本仓库位置 | 许可记录 |
|---|---|---|
| 冻结的ACT/LeRobot运行代码 | `runtime_lixin/` | 保留 `runtime_lixin/LICENSE` 及源码中的版权/许可证声明 |
| libfranka 0.20.4标准运动生成器 | `yin_controller/return_to_success_start/libfranka_0.20.4/` | 保留上游Apache-2.0许可与源码声明 |
| Yin ROS2控制器/状态流包 | `yin_controller/ros2_src/` | 保留各`package.xml`的许可证声明与源码头部 |
| 实验相机、接收器、训练及分析代码 | 对应原始目录 | 维持原有文件及声明；来源见`reproduction_manifest.json` |

上游项目链接：

- ACT：<https://github.com/tonyzhaozh/act>
- LeRobot：<https://github.com/huggingface/lerobot>
- libfranka：<https://github.com/frankarobotics/libfranka>

源文件保持成功快照的字节一致性。未添加许可证的文件不因公开收录而自动取得Apache、MIT等额外许可；再分发时需核对各文件的适用授权。
