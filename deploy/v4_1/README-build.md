# V4.1 离线包研发说明

此包以已冻结V4.1双分类器为基础，不是原V2检测器。照片判断采用整图分数与第二高局部分数的较小值；配置见models/model.json，模型来源、哈希与转换记录见models/export.json。app包含应用完整源码；developer/weights保留两个PT权重，供已有训练环境继续研发。

Windows运行依赖锁定在runtime-packages.txt；内置CPython 3.13.16（Windows x64）。项目历史pyproject.toml是研发/训练环境配置，不是Windows内置运行环境锁文件。内置环境只包含推理所需组件，不包含PyTorch/训练工具。

构建源码位于developer/source。构建主入口scripts/build_windows_v4_1.py，要求构建机Python、packaging、pefile、Zig 0.15.2及已下载的官方Windows运行时和Windows wheel文件。使用--help查看参数。Windows native C启动器和安装器用Zig交叉编译，均为真实Windows x64 PE程序；这不等于已完成Windows运行验证。

ZIP与Setup.exe包含相同内容。Setup在当前用户LOCALAPPDATA目录解压并创建桌面快捷方式，不注册服务、不修改系统Python/PATH、不开启防火墙、不自动清理旧版本。首次运行可能安装随附的官方VC++组件，需要用户确认。

运行：双击Mikania.exe。研发自检：Mikania.exe --self-test（失败时仍有可见错误窗口）；无浏览器启动：Mikania.exe --no-browser。运行入口会清理当前进程继承的MIKANIA_*参数，校验双模型SHA256并严格载入冻结配置。端口范围18041–18050，只监听127.0.0.1。

验证记录：developer/validation/onnx-validation.json记录217张冻结数据回放及分层CPU PT对照；所有照片判断与历史GPU结果一致，但CPU/GPU分数存在小幅差异。不能将此数值验证称作新盲测精度。

许可：依赖许可证保留在runtime/site-packages各dist-info目录，Python许可证在runtime/python/LICENSE.txt。模型来自Ultralytics体系，继续分发或商用前需核对其适用许可；本包不授予第三方模型的额外商业权利。用户原始数据不在此运行包中。
