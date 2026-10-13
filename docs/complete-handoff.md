# 完整研发与部署交付说明

完整交付由 `scripts/build_complete_handoff.py` 生成到 `dist/mikania-complete-handoff-1.1.0`。由于原始数据、训练数据和辅助图片合计超过 9 GB，构建器将交付内容拆成多个不超过常见移动介质限制的 ZIP，而不是生成一个难以复制和恢复的超大文件。

交付范围包括：

- Windows、macOS、Linux 的无 Docker CPU 部署包；
- 数据处理、训练、推理和网页源代码；
- 实验配置、环境记录、指标、曲线、最佳权重和最终 ONNX 模型；
- `raw-data` 中的原始图片及 YOLO 标签；
- `datasets/mikania_detection_v2_0_1` 中可直接训练的最终数据集（包含修正标签、困难负样本与独立负样本划分）；
- `data` 中的辅助图片和待复核素材；
- 标注规范、系统技术流程及项目研发文档。

构建器不会打包虚拟环境、缓存、认证信息、重复的原始数据 ZIP，或与薇甘菊项目无关的文件。每个交付文件和归档内研发文件均生成 SHA256 校验记录。

运行：

```bash
uv run python scripts/build_complete_handoff.py
```

构建结束后，按照交付目录中的 `README.md` 使用或恢复项目。
