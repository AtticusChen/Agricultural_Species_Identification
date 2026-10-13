# 薇甘菊远程训练 Phase 1 启动报告

日期：2026-08-04  
远程路径：`/home/cx/Agricultural_Species_Identification`  
结论：训练环境、数据同步、实验日志和 YOLO26n 烟雾测试均已通过；正式基线仍受标签质控门约束。

## 1. 服务器与环境

| 项目 | 实际值 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA GeForce RTX 5090 × 6 |
| 单卡显存 | 32,109 MiB 可用于框架 |
| NVIDIA 驱动 | 580.76.05 |
| 项目环境 | `/home/cx/.conda/envs/mikania-detect` |
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| Torchvision | 0.28.0+cu130 |
| Ultralytics | 8.4.115 headless |

6 张 GPU 全部可见，CUDA 矩阵计算验证通过。项目使用独立环境，没有修改服务器 base Python 3.13 或已有 `cx` 环境。

## 2. 数据同步与完整性

- 同步数据体积：2,806,304,332 bytes。
- 图片：462 张，远程均为真实文件，不是本地符号链接。
- 标签：462 份，非空标签行 941 行。
- `summary.json` SHA-256：`0567af6d91fb7d5a9afa7f4700cedbacb04d93c953ffef7c9ac1c11c017df0d1`。
- `samples.csv` SHA-256：`38bb58513358e553df988e29e8a410e3b6657f4f17847a0e9c34d7e8fc073ca6`。
- 远程逐图校验结果：原始训练副本与 manifest 哈希不一致数为 0。

## 3. JPEG 可复现性修复

Ultralytics 首次扫描时发现部分 JPEG 最后两个字节不是标准 `FFD9`，并自动重新压缩了训练/验证副本。调查确认：

- 共 152 张：train 94、val 20、test_internal 38、test_external 0。
- 150 张来自 HUAWEI Mate 70，2 张来自 vivo X Fold3。
- 图片可被 Pillow 完整解码，问题是文件结尾格式，不是像素解码失败。
- S0 后已从本地恢复被框架重写的 114 张远程图片；恢复后原数据哈希不一致数为 0。

新增 `mikania-data sanitize-jpeg`，生成 `mikania_detection_v1_0_1`：只在副本末尾追加 `FFD9`，不做 JPEG 重编码。派生版本结果：

- 图片 462、检测框 941、划分完全不变。
- 无损补齐 152 张。
- 派生 manifest SHA-256：`1d5cde49fc20d4a6681036f54553a60605b2c19419ce516ff958954d4088c5e7`。
- 逐图哈希不一致数：0。
- 再次调用 Ultralytics 官方图片检查：修复提示 0，检查后哈希变化 0。

## 4. S0 烟雾测试

实验 ID：`S0_yolo26n_640_smoke`  
模型：YOLO26n，COCO 预训练  
输入：640  
数据：245 张训练图、61 张验证图  
随机种子：20260804  
GPU：0  
训练：10 epochs，batch 16，AMP

| 指标 | 最佳权重验证结果 |
|---|---:|
| Precision | 0.4139 |
| Recall | 0.1429 |
| mAP50 | 0.1488 |
| mAP50:95 | 0.0556 |
| 训练峰值显存 | 约 3.14 GB |

这些指标来自 10 轮烟雾测试，只用于证明训练能运行、损失下降且 AP 非零，不进入论文主表，也不能用于判断最终精度。训练曲线显示训练和验证损失总体下降；可视化确认坐标、类别名和框绘制正常。

## 5. 产物

- 结构化日志：`experiments/logs/S0_yolo26n_640_smoke/run.json`
- 最佳权重：`experiments/runs/S0_yolo26n_640_smoke/weights/best.pt`
- 最后一轮权重：`experiments/runs/S0_yolo26n_640_smoke/weights/last.pt`
- 训练曲线、PR/F1 曲线、混淆矩阵和验证预测：`experiments/runs/S0_yolo26n_640_smoke/`
- 本地预览：`artifacts/remote/S0_yolo26n_640_smoke/`

## 6. 下一训练门

在启动 B0–B4 正式对比前完成：

1. 复核 16 个覆盖面积不低于 90% 的检测框。
2. 处理 12 张缺标签图片。
3. 将 123 张候选负样本判定为“无薇甘菊、含薇甘菊、不确定”。

完成后冻结 `v1.2-hn`，再并行运行 YOLO26n/s、Faster R-CNN、RT-DETRv2 和 960 分辨率消融。
