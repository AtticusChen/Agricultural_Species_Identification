from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
PORTABLE = ROOT / "deploy" / "portable"
BUILT_ROOT = ROOT / "dist" / "mikania-portable-2.1.0"
BUILT_SYSTEM = BUILT_ROOT / "runtime"


def test_installers_bootstrap_managed_python_without_system_python() -> None:
    unix = (PORTABLE / "install_unix.sh").read_text(encoding="utf-8")
    windows = (PORTABLE / "install_windows.ps1").read_text(encoding="utf-8")

    assert 'uv_version="0.11.32"' in unix
    assert "astral.sh/uv/${uv_version}/install.sh" in unix
    assert "UV_UNMANAGED_INSTALL" in unix
    assert "UV_PYTHON_INSTALL_DIR" in unix
    assert "--managed-python" in unix
    assert "launcher.py\" --self-test" in unix
    assert "command -v python" not in unix

    assert "astral.sh/uv/$UvVersion/install.ps1" in windows
    assert '"0.11.32"' in windows
    assert "UV_UNMANAGED_INSTALL" in windows
    assert "UV_PYTHON_INSTALL_DIR" in windows
    assert '"--managed-python"' in windows
    assert '"--self-test"' in windows


def test_nontechnical_entry_files_are_present() -> None:
    required = {
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
    }

    assert required <= {path.name for path in PORTABLE.iterdir()}


def test_launcher_self_test_loads_packaged_onnx_model() -> None:
    assert (BUILT_SYSTEM / "launcher.py").is_file(), "build portable bundle first"
    completed = subprocess.run(
        [sys.executable, str(BUILT_SYSTEM / "launcher.py"), "--self-test"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "环境和模型检查通过" in completed.stdout


def test_built_bundle_exposes_source_and_both_model_formats() -> None:
    assert (BUILT_ROOT / "程序源码" / "src" / "mikania_app" / "main.py").is_file()
    assert (BUILT_ROOT / "程序源码" / "src" / "mikania_data" / "dataset.py").is_file()
    assert (BUILT_ROOT / "程序源码" / "src" / "mikania_training" / "runner.py").is_file()
    assert (BUILT_ROOT / "程序源码" / "scripts" / "build_portable_bundle.py").is_file()

    model_root = BUILT_ROOT / "模型文件" / "mikania-yolo26s-v2"
    assert (model_root / "model.onnx").stat().st_size > 30_000_000
    assert (model_root / "model.pt").stat().st_size > 15_000_000
    assert (BUILT_ROOT / "交付内容与文件清单.txt").is_file()
    assert not (BUILT_ROOT / "_system").exists()


def test_windows_scripts_use_compatible_paths_encodings_and_visible_errors() -> None:
    entry_names = (
        "01-一键安装-Windows.bat",
        "02-启动识别系统-Windows.bat",
        "03-启动局域网共享-Windows.bat",
        "04-Windows故障诊断.bat",
    )
    for name in entry_names:
        source_text = (PORTABLE / name).read_text(encoding="ascii")
        assert "runtime" in source_text
        assert "pause" in source_text.lower()

        built_bytes = (BUILT_ROOT / name).read_bytes()
        assert built_bytes.isascii()
        assert b"\r\n" in built_bytes
        assert b"\n" not in built_bytes.replace(b"\r\n", b"")

    powershell_bytes = (BUILT_SYSTEM / "install_windows.ps1").read_bytes()
    assert powershell_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in powershell_bytes
    assert not (BUILT_ROOT / "安装与运行脚本").exists()
