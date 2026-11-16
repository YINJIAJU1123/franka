# 模型与外部资产

## 成功模型

- 模型：run02 EMA。
- 训练步数：30,000。
- 训练数据：`block_pick_place_v3_no_ep6`，14段；排除源`episode_000006`。
- 文件：`runtime_lixin/checkpoint/act_policy_ema.pth`。
- 文件大小：206,773,010字节。
- SHA256：`d9319c1b1e6f329f8a31d739cfd0593069622b949f3544a7b2bc0231dc39c852`。

使用这份EMA；同为30,000步的非EMA检查点是另一份权重。`dataset_stats.pth`随代码保留，不能换成其他数据集的统计量。

## Git中保留及排除的内容

保留源码、配置、URDF、小型数据统计量、episode映射、训练loss日志、验证JSON及轨迹图。

下列内容留在实验者自己的可信离线归档中：

- EMA大权重。
- 原始视频与2.36GB训练缓存。
- 输入图像NPZ夹具、关节轨迹Parquet。
- 原始实机/回位CSV与控制台日志。
- Conda/ROS安装和libfranka共享库。

每个已归档但未入Git的文件都记录在根目录`external_assets_manifest.json`。其中标记`required_by_original_runtime_manifest=true`的资产，需要恢复后才能满足原部署的文件校验。原代码中的绝对路径及原项目保护清单仍保留，恢复文件不意味着跨机器部署检查已经通过。

## 从可信完整归档恢复

如果持有2026-11-12整理的完整复现包，解压后会有`snapshot/`目录：

```bash
python tools/restore_external_assets.py --snapshot /path/to/ACT_success_reproduction_20261112/snapshot
python tools/verify_repository.py --require-runtime-assets
```

恢复工具只复制本清单中SHA256吻合的外部文件，不覆盖不同内容的既有文件，不下载数据或启动ROS/机器人。默认恢复原运行清单所需资产；需要原始CSV/日志供分析时才增加`--include-logs`。

原始视频、训练缓存和系统环境未包含在该完整代码归档中，应从实验数据目录及原环境单独保留。当前已验证主机的数据路径是：

```text
/home/lixin/act_gripper/dataset/block_pick_place_v3_no_ep6
```

Git忽略规则会阻止上述大权重和原始输入进入普通提交。提供本说明不意味着仓库托管了模型下载服务。
