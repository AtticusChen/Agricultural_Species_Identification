# 薇甘菊智能识别项目

## 当前版本：V4.1离线测试版（2026-10-13）

已完成“照片是否含薇甘菊”的双模型识别及粗区域定位，支持地面/无人机照片、电脑端批量识别、原图对照、下载标注图和导出CSV。当前V4.1不是精确植株分割系统；下文的早期实验设计及V2/V3交付记录为历史背景。

- [下载安装包与模型](https://github.com/AtticusChen/Agricultural_Species_Identification/releases/tag/v4.1.0-test)
- [直接下载 Windows 安装包](https://github.com/AtticusChen/Agricultural_Species_Identification/releases/download/v4.1.0-test/Mikania-Setup-4.1.0.exe) · [下载 ZIP 免安装包](https://github.com/AtticusChen/Agricultural_Species_Identification/releases/download/v4.1.0-test/mikania-windows-v4.1.0.zip)
- Windows 10/11 x64：下载 `Mikania-Setup-4.1.0.exe` 双击安装，或完整解压 `mikania-windows-v4.1.0.zip` 后运行 `Mikania.exe`。
- 内含Python环境、两个ONNX模型、原始PT权重、完整网页、源码和图文手册。无需预装Python、Docker或连接VPN；首次运行可能需要确认安装附带的微软VC++运行组件。
- **测试交付版：尚未进行Windows实机安装验收，未做商业代码签名。** 已完成模型数值回放、网页功能与包完整性检查；详细见[发布记录](docs/v4_1-release-20261013.md)。
- 本仓库按所有者要求公开，安装包可直接下载，无需仓库授权。安装包、权重放在Releases；原始训练图片、原始标注、个人凭据和未发布专利材料不在Git中，仅保留网页展示所需的处理后示例图片。

V4.1提高检出率的同时存在误报：已有地面74张评估中TP54/FN2/FP4/TN14，阴性误报4/18（22.2%）；外部阳性照片检出98/100，无人机阳性照片检出42/43。后两组没有阴性，不能视为总体准确率。这些是已使用过的评估/开发数据，不是新增盲测。绿色区域仅供粗定位，识别结果需人工复核。

### 代码目录

- `src/`：网页后端、前端、推理、数据整理和训练工具。
- `configs/`：数据划分与实验配置；`tests/`：回归测试。
- `deploy/v4_1/`：精确冻结配置、Windows原生启动器/安装器源码、离线入口和手册源文件。
- `scripts/`：训练评估、模型导出、构建与部署脚本；`docs/`：设计与结果记录。
- `model_store/mikania-ensemble-v4.1/export.json`：模型来源、转换配置及SHA256摘要；二进制权重从Release下载。

### 研发者获取权重

登录GitHub CLI并克隆仓库后，可运行：

```bash
gh release download v4.1.0-test \
  --repo AtticusChen/Agricultural_Species_Identification \
  --pattern '*.onnx' --pattern '*.pt' \
  --dir model_store/mikania-ensemble-v4.1
```

仓库不包含原始数据、旧版dist包或本地完整实验产物。历史资产集成测试需要恢复相应文件才能运行；不依赖大文件的源码测试可按[GitHub交付说明](docs/github-delivery.md)执行。训练环境与Windows内置推理环境分开管理，历史`pyproject.toml`不是Windows运行包的依赖锁文件。

## 项目研发背景（历史阶段）

本仓库用于研发两条互补任务：

- 手机/相机近景图：薇甘菊单类目标检测；
- 无人机 RGB 图：薇甘菊冠层斑块语义分割与覆盖分析。

最初的地面检测Phase 1采用按采集组和近重复簇隔离的数据划分；后续已推进至V4.1照片级双模型识别。无人机精确斑块分割和覆盖分析仍是后续方向，不是本次已完成能力。

项目最终交付目标是可跨设备使用的薇甘菊识别系统：远程服务器负责训练，冻结模型通过网页提供图片检测，并支持无需 Docker 的笔记本 CPU 部署。总体设计见 [系统方案 v1.0](docs/system-architecture-v1.0.md)。

## 本地环境

```bash
uv sync --extra dev
uv run pytest
```

生成只读数据审计：

```bash
uv run mikania-data audit \
  --data data \
  --output artifacts/audit_v1
```

生成 50 张近景图和 30 张无人机图的试标包：

```bash
uv run mikania-data pilot \
  --data data \
  --config configs/pilot_v1.json \
  --output artifacts/pilot_v1

uv run mikania-data preview \
  --manifest artifacts/pilot_v1/manifest.csv \
  --pilot-root artifacts/pilot_v1 \
  --output artifacts/pilot_v1_preview
```

试标包默认使用指向原图的符号链接，不复制或修改原始图片。详细操作见 [试标协议](docs/annotation-pilot-protocol.md)。

从 `raw-data` 生成规范化、按采集组隔离的地面检测数据集：

```bash
uv run mikania-data build-detection \
  --raw-data raw-data \
  --negative-data data \
  --config configs/detection_dataset_v1.json \
  --output datasets/mikania_detection_v1
```

原始图片不会被修改。已有标签进入训练数据结构；缺标图片和未确认近似种分别进入待复核队列。

为避免训练框架自动重新压缩缺少标准 JPEG 结束标记的手机图片，可生成无损派生版本：

```bash
mikania-data sanitize-jpeg \
  --dataset datasets/mikania_detection_v1 \
  --output datasets/mikania_detection_v1_0_1 \
  --version mikania_detection_v1.0.1-jpeg-eoi
```

该操作只在副本末尾追加 `FFD9`，不重新编码像素，并在 manifest 中记录每张修复图片的前后哈希。

## 关键研发约束

- 不把相邻连拍图或同一无人机航次切片随机分散到训练集和测试集；
- 近似种图片必须经人工确认后才能作为空标签负样本；
- 近景主类别保持单类，叶、花、茎和物候作为辅助属性；
- 无人机图优先标斑块掩膜，而不是人为制造“单株”矩形框；
- 所有实验记录数据版本、分组版本、模型版本、随机种子和阈值。

## 远程训练

正式训练服务器路径为 `/home/cx/Agricultural_Species_Identification`。服务器环境与部署原则见 [远程训练环境](docs/remote-training-environment.md)。认证信息不得写入仓库。

安装训练依赖后，用结构化配置运行实验：

```bash
mikania-train \
  --project-root /home/cx/Agricultural_Species_Identification \
  --config configs/experiments/S0_yolo26n_640_smoke.json
```

运行前会保存配置和数据摘要哈希，运行后将指标、环境和检查点路径写入 `experiments/logs/<experiment_id>/run.json`。同名实验不会被覆盖。

## 识别页面

### 2026-09 电脑端工作台改版

新版已于2026-09-21正式上线，提供批量识别、原图/结果对比、大图查看、图片下载、会话记录筛选和CSV导出，使用指南集成在页面中。按用户要求仅针对电脑和笔记本优化，不制作手机专用布局。当时服务器保持V3模型；2026-10-13按用户选择切换到V4.1，见本文顶部当前版本说明。

网页改版当时的状态、真实验证与备份说明见 [网页改版记录](docs/web-redesign-release.md)。以下v2.1.0离线包是历史交付记录，当前安装包应使用顶部链接中的V4.1。

### 笔记本便携版（无需 Docker）

面向非技术用户的交付包位于 `dist/mikania-portable-2.1.0.zip`。解压后根目录可直接看到完整程序源码、ONNX 推理模型、PT 训练权重、示例图、逐步操作手册，以及 Windows、macOS、Linux 的一键安装和启动入口。目标电脑不需要预装 Python；安装器会自动配置项目专用运行环境。首次安装需要网络，完成后可离线识别。Windows 运行组件使用纯英文 `runtime` 路径，入口失败时会保留窗口并显示日志位置。v2.1.0 默认使用 D1 YOLO26s 模型、15% 置信度和“精细扫描”，并保留速度更快的“快速扫描”模式。

具体双击步骤和局域网访问方式见包内 `README.md`。便携包已在 macOS ARM64/Python 3.12 和 Ubuntu x86_64/Python 3.10 上完成独立解压、安装、启动和图片上传测试。

### 当前统一入口

在本项目的研发 Mac 上双击根目录的 `打开薇甘菊识别.command`，或运行：

```bash
./打开薇甘菊识别.command
```

脚本会建立到训练服务器的安全转发并打开 `http://127.0.0.1:18000`（仅此Mac）。电脑重启后重新双击入口，首次连接可能需要服务器账号密码；输入时不显示字符，输入完按回车。2026-10-13已修正为直接提示服务器密码，不再先要求本机SSH私钥口令；脚本不保存密码。它会探测到服务器的可用连接路径，不修改全局VPN/代理配置。不要直接打开 `static/index.html` 网页源文件。

同一内网或已连接相同 VPN 且网络路径可用的其他设备也可访问 `http://10.33.231.250:8000`；`10.33.*` 是私网地址，不在该网络中的设备无法直接访问。

服务器使用 `mikania-app.service` 常驻运行，服务定义位于 `deploy/systemd/mikania-app.service`。

### 开发启动

安装页面与推理依赖：

```bash
uv sync --extra app --extra dev
```

指定 v2 部署模型并启动页面：

```bash
MIKANIA_MODEL_PATH=model_store/mikania-yolo26s-v2/model.onnx MIKANIA_DEVICE=cpu \
  uv run uvicorn mikania_app.main:app --host 0.0.0.0 --port 8000
```

浏览器打开 `http://<服务器地址>:8000`，上传 JPG、PNG 或 WebP 图片。只要模型检出至少一个薇甘菊区域，页面就显示“识别到薇甘菊”并标出位置；否则显示“未识别到薇甘菊”。
