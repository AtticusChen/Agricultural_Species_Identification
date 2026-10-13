# GitHub交付说明

账号：AtticusChen。仓库：Agricultural_Species_Identification。2026-10-13按所有者明确要求由私有改为公开，其他人可直接下载Releases中的安装包，无需单独授予仓库访问权限。原始数据、密码及未发布专利材料仍不上传。

## 上传范围

- Git：源码、配置、测试、设计/评估文档、V4.1模型元数据和网页示例。
- Release `v4.1.0-test`：Windows Setup、免安装ZIP、SHA256清单、两份ONNX与两份PT权重、模型导出记录及验证摘要。
- 未上传：原始图片与原始标注、派生训练集、完整实验缓存、旧交付包、本地环境、密码/密钥、专利工作目录及无关压缩包。本次不是全部原始研发数据的云备份。

## Windows运行

从Releases下载Setup.exe安装，或完整解压ZIP后双击Mikania.exe。不要只下载GitHub的“Source code.zip”后期待里面包含模型及Windows运行环境；它只包含Git跟踪的源码。图文使用说明在安装包根目录。

测试版尚未完成Windows实机安装验收，系统可能提示未知发布者。先核实来源和SHA256，勿关闭安全保护。模型结果需人工复核，不可把绿色粗区域视为精确分割。

## 源码测试

使用Python 3.11/3.12研发环境安装`app`和`dev`依赖后，以下测试不要求完整历史交付包：

```bash
uv sync --extra app --extra dev
uv run pytest --ignore=tests/test_portable_bundle.py --ignore=tests/test_web_deploy.py
node --test tests/test_workspace_core.mjs
```

`test_portable_bundle.py`验证历史V2包，`test_web_deploy.py`含V3归档哈希校验；部分示例溯源及模型集成检查在缺少本地manifest/权重时会自动跳过。源码检查与完整交付集成检查的边界需区分，不得把跳过历史资产测试说成全部集成验收。

原开发目录在发布前141个Python测试和10个JavaScript测试通过；另有29张服务器接口回归、217张ONNX数值回放及真实网页功能测试。测试状态不意味着现场识别达到完全可靠，也不替代Windows实机验收。

2026-10-13另外从Git暂存区导出仅源码副本（不带原始数据、模型和历史产物），按上述命令实测：109个Python测试通过、4个依赖本地产物的检查跳过，10个JavaScript测试通过。仅源码副本没有偷偷读取完整数据来冒充可复现。

## 许可证与原始资料

公开仓库供查看及下载安装包；此次只调整可见性，不自动为项目添加或变更开源许可证。第三方Python依赖许可证随运行包保留；模型与Ultralytics体系相关的适用许可在再分发/商用前应另行核对。科研原始数据与专利材料继续保留在原有本地/服务器位置。
