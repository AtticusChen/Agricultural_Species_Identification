# 薇甘菊 V4.1 Windows 离线测试版

适用Windows 10/11 x64；CPU离线识别，无需Python、Docker、VPN或训练服务器。

## 普通用户下载

1. 推荐 `Mikania-Setup-4.1.0.exe`：双击安装，创建桌面快捷方式。
2. 备用 `mikania-windows-v4.1.0.zip`：完整解压后运行 `Mikania.exe`。
3. `SHA256SUMS-v4.1.0.txt`：核对安装包/ZIP摘要。

两种运行包内容相同：运行环境、双ONNX模型、原始PT权重、网页、源码、示例和图文手册。首次缺少微软VC++组件时会提示安装随附的官方组件，可能需要管理员确认。

不要把GitHub自动生成的Source code.zip当作离线运行包。原始训练图片和标注没有放入本次Release。

## 研发附件

`scene.onnx`/`patch.onnx`为推理模型；`scene.pt`/`patch.pt`为原始权重；`export.json`记录模型来源、参数和模型SHA256；`verification-summary.json`记录检查范围。

## 验证边界

- 217张冻结数据数值回放，ONNX与历史GPU照片判断一致；29张正式服务API回归通过。
- 已完成真实网页功能、源码测试及安装包结构/摘要检查。
- **尚未执行Windows实机安装与运行验收，未做商业代码签名，属于预发布测试包。**
- V4.1偏重检出率，已有地面阴性误报4/18（22.2%）；绿色区域是粗定位，不是精确边界，所有判断应人工复核。

如运行失败，请提供系统版本、错误截图与`%LOCALAPPDATA%\Mikania\4.1\logs`中的日志。安装错误另查`%TEMP%\mikania-install-error.txt`。
