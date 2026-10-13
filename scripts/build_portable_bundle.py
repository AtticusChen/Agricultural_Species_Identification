from __future__ import annotations

import hashlib
import shutil
import stat
import zipfile
from pathlib import Path

VERSION = "2.1.0"
MODEL_ID = "mikania-yolo26s-v2"
ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = ROOT / "dist"
BUNDLE_NAME = f"mikania-portable-{VERSION}"
BUNDLE_ROOT = DIST_ROOT / BUNDLE_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )


def _normalize_windows_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    windows_text = text.replace("\n", "\r\n")
    if path.suffix.lower() == ".bat":
        if not windows_text.isascii():
            raise ValueError(f"Windows batch files must contain ASCII only: {path}")
        path.write_bytes(windows_text.encode("ascii"))
    elif path.suffix.lower() == ".ps1":
        path.write_bytes(b"\xef\xbb\xbf" + windows_text.encode("utf-8"))


def _safe_replace_directory(path: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != DIST_ROOT.resolve() or resolved.name != BUNDLE_NAME:
        raise ValueError(f"Refusing to replace unexpected directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def build() -> tuple[Path, Path]:
    DIST_ROOT.mkdir(exist_ok=True)
    _safe_replace_directory(BUNDLE_ROOT)
    BUNDLE_ROOT.mkdir()

    portable = ROOT / "deploy" / "portable"
    model = ROOT / "model_store" / MODEL_ID
    source_root = BUNDLE_ROOT / "程序源码"
    runtime_root = BUNDLE_ROOT / "runtime"
    source_root.mkdir()
    runtime_root.mkdir()

    for source_dir in ("src", "scripts", "configs", "tests"):
        _copy_tree(ROOT / source_dir, source_root / source_dir)
    for source_file in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(ROOT / source_file, source_root / source_file)

    packaged_model = BUNDLE_ROOT / "模型文件" / MODEL_ID
    packaged_model.mkdir(parents=True)
    for model_file in ("model.onnx", "model.pt", "metadata.json", "model_card.md"):
        shutil.copy2(model / model_file, packaged_model / model_file)
    for internal_file in (
        "launcher.py",
        "requirements.txt",
        "install_windows.bat",
        "install_windows.ps1",
        "install_macos.command",
        "install_linux.sh",
        "install_unix.sh",
        "start_windows.bat",
        "start_macos.command",
        "start_linux.sh",
        "README.md",
    ):
        target = runtime_root / internal_file
        shutil.copy2(portable / internal_file, target)
        if target.suffix.lower() in {".bat", ".ps1"}:
            _normalize_windows_script(target)
        if target.suffix in {".command", ".sh"}:
            target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    for entry_file in (
        "00-请先阅读-快速开始.txt",
        "01-一键安装-Windows.bat",
        "02-启动识别系统-Windows.bat",
        "03-启动局域网共享-Windows.bat",
        "04-Windows故障诊断.bat",
        "01-一键安装-macOS.command",
        "02-启动识别系统-macOS.command",
        "03-启动局域网共享-macOS.command",
        "01-一键安装-Linux.sh",
        "02-启动识别系统-Linux.sh",
        "03-启动局域网共享-Linux.sh",
    ):
        target = BUNDLE_ROOT / entry_file
        shutil.copy2(portable / entry_file, target)
        if target.suffix.lower() == ".bat":
            _normalize_windows_script(target)
        if target.suffix in {".command", ".sh"}:
            target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    for manual_name in (
        "薇甘菊识别系统-零基础操作手册.docx",
        "薇甘菊识别系统-零基础操作手册.pdf",
    ):
        manual = ROOT / "docs" / manual_name
        if not manual.is_file():
            raise FileNotFoundError(f"Missing required user manual: {manual}")
        shutil.copy2(manual, BUNDLE_ROOT / manual_name)

    example_source = (
        ROOT
        / "datasets"
        / "mikania_detection_v2_0_1"
        / "images"
        / "test_internal"
        / "IMG_20251118_165528.jpg"
    )
    examples = BUNDLE_ROOT / "示例图片"
    examples.mkdir()
    shutil.copy2(example_source, examples / "薇甘菊示例.jpg", follow_symlinks=True)
    (runtime_root / "VERSION").write_text(VERSION + "\n", encoding="utf-8")

    onnx_path = packaged_model / "model.onnx"
    pt_path = packaged_model / "model.pt"
    inventory = f"""薇甘菊识别系统 v{VERSION}｜交付内容与文件清单
================================================

本包已经包含可运行代码、模型权重和一键安装脚本。解压后可直接看到：

一、程序源码
  程序源码/src/mikania_app       网页与 ONNX 推理代码
  程序源码/src/mikania_data      数据整理、校验与预览代码
  程序源码/src/mikania_training  YOLO 训练与导出代码
  程序源码/scripts               研发、打包和环境检查脚本
  程序源码/configs               实验与训练配置
  程序源码/tests                 自动化测试
  程序源码/pyproject.toml        项目依赖配置
  程序源码/uv.lock               可复现依赖锁定文件

二、模型文件
  模型文件/{MODEL_ID}/model.onnx
    用途：普通电脑网页识别使用
    大小：{onnx_path.stat().st_size:,} 字节
    SHA256：{_sha256(onnx_path)}
  模型文件/{MODEL_ID}/model.pt
    用途：继续训练、评估和重新导出使用
    大小：{pt_path.stat().st_size:,} 字节
    SHA256：{_sha256(pt_path)}
  同目录 metadata.json 和 model_card.md 记录模型版本、类别和使用说明。

三、安装与运行
  runtime（安装与运行脚本）目录保存环境安装器、启动器、依赖清单和安装日志。
  普通用户只需按电脑系统双击根目录中以 01、02、03 开头的文件。

四、示例与手册
  示例图片                    首次识别测试图片
  薇甘菊识别系统-零基础操作手册  Word 和 PDF 操作手册

五、数据范围说明
  为了让便携包便于在笔记本之间复制，本 ZIP 不重复装入约 10 GB 的原始图片、
  标注数据和训练运行结果。完整研发数据应使用独立的“完整研发交付包”；本便携包
  已包含运行、继续开发和模型复用所需的源码与两种模型权重。

六、完整性校验
  根目录“交付文件SHA256.txt”记录包内文件的校验值。
"""
    (BUNDLE_ROOT / "交付内容与文件清单.txt").write_text(inventory, encoding="utf-8")

    files = sorted(
        path
        for path in BUNDLE_ROOT.rglob("*")
        if path.is_file() and path.name != "交付文件SHA256.txt"
    )
    checksum_lines = [f"{_sha256(path)}  {path.relative_to(BUNDLE_ROOT)}" for path in files]
    (BUNDLE_ROOT / "交付文件SHA256.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    archive = DIST_ROOT / f"{BUNDLE_NAME}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zip_file:
        for path in sorted(BUNDLE_ROOT.rglob("*")):
            if path.is_file():
                zip_file.write(path, Path(BUNDLE_NAME) / path.relative_to(BUNDLE_ROOT))

    return BUNDLE_ROOT, archive


if __name__ == "__main__":
    bundle, zip_path = build()
    print(f"Bundle: {bundle}")
    print(f"Archive: {zip_path}")
    print(f"Archive SHA256: {_sha256(zip_path)}")
