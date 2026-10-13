"""Build a Windows x64 offline ZIP and native self-extracting installer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
NAME = "mikania-windows-v4.1.0"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wheel_inventory(paths: list[Path]) -> list[dict]:
    env = default_environment()
    env.update(
        sys_platform="win32",
        os_name="nt",
        platform_system="Windows",
        platform_machine="AMD64",
        python_version="3.13",
        python_full_version="3.13.16",
        extra="",
    )
    packages = {}
    for path in paths:
        _, _, _, tags = parse_wheel_filename(path.name)
        if not any(t.platform in {"any", "win_amd64"} for t in tags):
            raise ValueError(f"Not a Windows x64 wheel: {path}")
        with ZipFile(path) as archive:
            metadata = BytesParser().parsebytes(
                archive.read(
                    next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
                )
            )
        key = canonicalize_name(metadata["Name"])
        if key in packages:
            raise ValueError(f"Duplicate wheel: {key}")
        packages[key] = {
            "name": key,
            "version": metadata["Version"],
            "file": path.name,
            "sha256": sha(path),
            "requirements": metadata.get_all("Requires-Dist", []),
        }
    for package in packages.values():
        for raw in package["requirements"]:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate(env):
                dependency = packages.get(canonicalize_name(requirement.name))
                if dependency is None or dependency["version"] not in requirement.specifier:
                    raise ValueError(f"Unresolved Windows dependency: {package['name']}: {raw}")
    return sorted(packages.values(), key=lambda value: value["name"])


def unpack(path: Path, target: Path, wheel: bool = False) -> None:
    with ZipFile(path) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if not parts or member.is_dir():
                continue
            if ".." in parts or parts[0].startswith("/"):
                raise ValueError("Unsafe archive path")
            if wheel and parts[0].endswith(".data"):
                if parts[1] not in {"purelib", "platlib"}:
                    raise ValueError(
                        f"Wheel requires unsupported installation scheme: {member.filename}"
                    )
                parts = parts[2:]
            destination = target.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError(f"Duplicate archive entry: {destination}")
            with archive.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--zig", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    folder = args.output / NAME
    zip_path = args.output / f"{NAME}.zip"
    setup_path = args.output / "Mikania-Setup-4.1.0.exe"
    if folder.exists() or zip_path.exists() or setup_path.exists():
        raise SystemExit("Output already exists; use a fresh directory. Nothing overwritten.")
    wheels = sorted((args.downloads / "wheels").glob("*.whl"))
    packages = wheel_inventory(wheels)
    export = json.loads((args.models / "export.json").read_text())
    for branch in ("scene", "patch"):
        assert sha(args.models / f"{branch}.onnx") == export["models"][branch]["sha256"]
        assert sha(args.models / f"{branch}.pt") == export["config"][f"{branch}_pt_sha256"]
    assert json.loads((ROOT / "deploy/v4_1/model.json").read_text()) == export["config"]
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    shutil.copytree(ROOT / "src/mikania_app", folder / "app/mikania_app", ignore=ignore)
    unpack(args.downloads / "python-3.13.16-embed-amd64.zip", folder / "runtime/python")
    (folder / "runtime/python/python313._pth").write_text(
        "python313.zip\n.\n../site-packages\n../../app\nimport site\n", encoding="utf-8"
    )
    for wheel in wheels:
        unpack(wheel, folder / "runtime/site-packages", wheel=True)
    copy(args.downloads / "VC_redist.x64.exe", folder / "runtime/VC_redist.x64.exe")
    copy(ROOT / "deploy/v4_1/launcher.py", folder / "runtime/launcher.py")
    for name in ("scene.onnx", "patch.onnx", "export.json"):
        copy(args.models / name, folder / "models" / name)
    copy(ROOT / "deploy/v4_1/model.json", folder / "models/model.json")
    for branch in ("scene", "patch"):
        copy(args.models / f"{branch}.pt", folder / "developer/weights" / f"{branch}.pt")
    for directory in ("src", "scripts", "tests", "configs", "deploy/v4_1"):
        shutil.copytree(ROOT / directory, folder / "developer/source" / directory, ignore=ignore)
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        copy(ROOT / name, folder / "developer/source" / name)
    for name in ("onnx-validation.json", "preview-validation.json", "preview-cpu-validation.json"):
        copy(args.validation / name, folder / "developer/validation" / name)
    for name in ("production-validation.json", "verification-summary.json"):
        if (args.validation / name).exists():
            copy(args.validation / name, folder / "developer/validation" / name)
    for phase in ("browser-production", "browser-offline"):
        if (args.validation / phase / "report.json").exists():
            copy(
                args.validation / phase / "report.json",
                folder / "developer/validation" / f"{phase}.json",
            )
    for name in ("home.png", "comparison.png"):
        copy(args.validation / "browser-offline" / name, folder / "使用说明资源" / name)
    copy(ROOT / "deploy/v4_1/README-build.md", folder / "developer/研发与重建说明.md")
    copy(ROOT / "deploy/v4_1/使用说明.html", folder / "使用说明.html")
    (folder / "00-开始使用.txt").write_text(
        "薇甘菊 V4.1 · Windows 10/11 x64 离线测试包\n\n"
        "1. 完整解压整个ZIP，双击 Mikania.exe。无需安装Python或连接VPN。\n"
        "2. 首次缺少微软运行组件时，按提示安装包内组件，可能需要管理员确认。\n"
        "3. 等待浏览器自动打开；保持程序窗口开启。查看详细步骤：使用说明.html。\n"
        "4. 本版尚待Windows实机验收，建议先使用网页示例测试。\n"
        "5. 识别结果需人工复核，绿色区域不是精确边界；地面误报率4/18。\n"
        "日志：%LOCALAPPDATA%\\Mikania\\4.1\\logs\n"
        "含运行环境、模型与研发代码，不含全部原始训练图片和标注。\n",
        encoding="utf-8-sig",
    )
    (folder / "developer/runtime-packages.txt").write_text(
        "\n".join(f"{p['name']}=={p['version']}" for p in packages) + "\n", encoding="utf-8"
    )
    common = [
        str(args.zig.resolve()),
        "cc",
        "-target",
        "x86_64-windows-gnu",
        "-municode",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]
    subprocess.run(
        [
            *common,
            str(ROOT / "deploy/v4_1/windows_launcher.c"),
            "-o",
            str(folder / "Mikania.exe"),
            "-luser32",
            "-lshell32",
            "-lkernel32",
        ],
        check=True,
    )
    subprocess.run(
        [
            *common,
            str(ROOT / "deploy/v4_1/windows_installer.c"),
            "-o",
            str(setup_path),
            "-luser32",
            "-lshell32",
            "-lkernel32",
        ],
        check=True,
    )
    download_files = [p for p in args.downloads.iterdir() if p.is_file()]
    manifest = {
        "version": "4.1.0",
        "platform": "Windows 10/11 x64",
        "windows_execution_tested": False,
        "python": "3.13.16 embedded amd64",
        "models": export,
        "packages": packages,
        "downloads": [{"name": p.name, "sha256": sha(p)} for p in download_files],
        "sources": {
            "python": "https://www.python.org/ftp/python/3.13.16/",
            "vc_runtime": "https://aka.ms/vc14/vc_redist.x64.exe",
            "wheels": "https://pypi.org/",
            "compiler": "https://ziglang.org/download/",
        },
        "files": {
            str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()
        },
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with ZipFile(zip_path, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(args.output))
    with ZipFile(zip_path) as archive:
        assert archive.testzip() is None
    with setup_path.open("ab") as target, zip_path.open("rb") as source:
        shutil.copyfileobj(source, target)
        target.write(b"MIKANIA1" + struct.pack("<Q", zip_path.stat().st_size))
    checksums = args.output / "SHA256SUMS-v4.1.0.txt"
    checksums.write_text("\n".join(f"{sha(p)}  {p.name}" for p in (setup_path, zip_path)) + "\n")
    print(
        json.dumps(
            {
                "folder": str(folder),
                "installer": str(setup_path),
                "zip": str(zip_path),
                "installer_bytes": setup_path.stat().st_size,
                "packages": len(packages),
                "windows_execution_tested": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
