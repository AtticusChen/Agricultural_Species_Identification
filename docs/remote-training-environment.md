# 远程训练环境

日期：2026-08-04

## 已核验资源

- 操作系统：Ubuntu 22.04.5 LTS
- GPU：NVIDIA GeForce RTX 5090 × 6
- 单卡显存：约 32 GB
- 驱动：580.76.05
- 驱动支持的 CUDA：13.0
- `/home/cx` 所在文件系统可用空间：约 1.9 TB
- 项目路径：`/home/cx/Agricultural_Species_Identification`
- 项目代码、训练数据和实验配置已同步。

## 已冻结训练环境

- Conda 前缀：`/home/cx/.conda/envs/mikania-detect`
- Python：3.11.15
- PyTorch：2.13.0+cu130
- Torchvision：0.28.0+cu130
- Ultralytics headless：8.4.115
- CUDA 可用性：通过，6 张 GPU 均可见，计算能力 12.0
- 环境快照：`environment/remote-training-env.json`
- 完整依赖：`environment/remote-pip-freeze.txt`

## 部署原则

- 远程服务器仅用于训练、批量评估和模型导出；本地负责代码研发、数据审计和界面开发。
- 不使用 base 环境的 Python 3.13 直接安装研究依赖；创建独立 Python 3.11 环境。
- 第一轮基线默认单卡运行，确认显存、吞吐和可复现性后再使用多卡。
- 原始数据、标注、划分文件和代码分别同步，并在远程校验文件数量及哈希。
- 训练任务固定 GPU 编号、随机种子、代码版本、数据版本和输出目录。
- 密码、令牌、SSH 私钥不得写入仓库、脚本、实验日志或模型配置。

## 当前执行状态

- 远程训练数据为 462 张真实文件、462 份标签和 941 个目标框；核心清单哈希与本地一致。
- 已完成 `S0_yolo26n_640_smoke`，训练入口、CUDA、数据加载、检查点和结构化日志均通过。
- 正式基线使用 `mikania_detection_v1_0_1` 派生数据；该版本无损补齐 152 张 JPEG 的结束标记，避免框架自动重新压缩图片。
- B0/B1/B4 配置已建立，但要等标签质控与困难负样本确认后进入正式对比。
