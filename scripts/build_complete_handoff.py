from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

VERSION = "1.1.0"
PORTABLE_VERSION = "2.1.0"
ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = ROOT / "dist"
HANDOFF_NAME = f"mikania-complete-handoff-{VERSION}"
HANDOFF_ROOT = DIST_ROOT / HANDOFF_NAME
PROJECT_PREFIX = PurePosixPath("Agricultural_Species_Identification")
MAX_AUXILIARY_PART_BYTES = 2_500_000_000
BLOCK_SIZE = 4 * 1024 * 1024

COMPRESSED_SUFFIXES = {
    ".7z",
    ".avi",
    ".docx",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp4",
    ".onnx",
    ".pdf",
    ".png",
    ".pt",
    ".webp",
    ".zip",
}
SKIP_DIRECTORY_NAMES = {
    ".claude",
    ".pytest_cache",
    ".ruff_cache",
    ".ultralytics",
    ".venv",
    "__pycache__",
}
SKIP_FILE_NAMES = {".DS_Store", "mikania_app.pid"}
SKIP_SUFFIXES = {".cache", ".pyc", ".pyo"}


@dataclass(frozen=True)
class ArchiveEntry:
    source: Path
    relative_path: Path


@dataclass(frozen=True)
class FileRecord:
    archive: str
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ArchiveRecord:
    filename: str
    size_bytes: int
    file_count: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _should_skip(path: Path) -> bool:
    return (
        path.name in SKIP_FILE_NAMES
        or path.suffix.lower() in SKIP_SUFFIXES
        or any(part in SKIP_DIRECTORY_NAMES for part in path.parts)
    )


def _regular_files(path: Path) -> list[ArchiveEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Required handoff source is missing: {path}")
    entries: list[ArchiveEntry] = []
    candidates = [path] if path.is_file() else sorted(path.rglob("*"))
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_file() or _should_skip(candidate):
            continue
        entries.append(ArchiveEntry(candidate, candidate.relative_to(ROOT)))
    return entries


def _safe_replace_handoff() -> None:
    resolved = HANDOFF_ROOT.resolve()
    if resolved.parent != DIST_ROOT.resolve() or resolved.name != HANDOFF_NAME:
        raise ValueError(f"Refusing to replace unexpected directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _compression_for(path: Path) -> int:
    if path.suffix.lower() in COMPRESSED_SUFFIXES:
        return zipfile.ZIP_STORED
    return zipfile.ZIP_DEFLATED


def _write_entry(
    archive: zipfile.ZipFile,
    archive_name: str,
    entry: ArchiveEntry,
) -> FileRecord:
    member_name = str(PROJECT_PREFIX / PurePosixPath(entry.relative_path.as_posix()))
    info = zipfile.ZipInfo.from_file(entry.source, member_name)
    info.compress_type = _compression_for(entry.source)
    digest = hashlib.sha256()
    size = 0
    with entry.source.open("rb") as source, archive.open(
        info,
        mode="w",
        force_zip64=True,
    ) as destination:
        for block in iter(lambda: source.read(BLOCK_SIZE), b""):
            destination.write(block)
            digest.update(block)
            size += len(block)
    return FileRecord(
        archive=archive_name,
        relative_path=entry.relative_path.as_posix(),
        size_bytes=size,
        sha256=digest.hexdigest(),
    )


def _build_archive(
    name: str,
    entries: list[ArchiveEntry],
) -> tuple[ArchiveRecord, list[FileRecord]]:
    destination = HANDOFF_ROOT / name
    temporary = HANDOFF_ROOT / f".{name}.tmp"
    if temporary.exists():
        temporary.unlink()
    total_bytes = sum(entry.source.stat().st_size for entry in entries)
    print(f"Building {name}: {len(entries)} files, {total_bytes / 1_000_000_000:.2f} GB")
    records: list[FileRecord] = []
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for entry in entries:
            records.append(_write_entry(archive, name, entry))
    os.replace(temporary, destination)
    archive_record = ArchiveRecord(
        filename=name,
        size_bytes=destination.stat().st_size,
        file_count=len(records),
        sha256=_sha256(destination),
    )
    print(f"Finished {name}: {archive_record.size_bytes / 1_000_000_000:.2f} GB")
    return archive_record, records


def _partition(entries: list[ArchiveEntry], maximum_bytes: int) -> list[list[ArchiveEntry]]:
    parts: list[list[ArchiveEntry]] = []
    current: list[ArchiveEntry] = []
    current_bytes = 0
    for entry in entries:
        size = entry.source.stat().st_size
        if current and current_bytes + size > maximum_bytes:
            parts.append(current)
            current = []
            current_bytes = 0
        current.append(entry)
        current_bytes += size
    if current:
        parts.append(current)
    return parts


def _code_entries() -> list[ArchiveEntry]:
    sources = [
        ROOT / ".gitignore",
        ROOT / ".python-version",
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "薇甘菊数据标注规范_学生版.docx",
        ROOT / "薇甘菊智能识别系统_识别方法与技术流程.docx",
        ROOT / "artifacts",
        ROOT / "configs",
        ROOT / "deploy",
        ROOT / "docs",
        ROOT / "experiments",
        ROOT / "model_store",
        ROOT / "scripts",
        ROOT / "src",
        ROOT / "tests",
        ROOT / "datasets" / "mikania_detection_v2_0_1" / "manifests",
    ]
    entries: list[ArchiveEntry] = []
    for source in sources:
        entries.extend(_regular_files(source))
    return sorted(entries, key=lambda item: item.relative_path.as_posix())


def _handoff_readme(auxiliary_parts: int) -> str:
    auxiliary_names = "\n".join(
        f"   - `04-auxiliary-data-part{part:02d}.zip`"
        for part in range(1, auxiliary_parts + 1)
    )
    return f"""# 薇甘菊项目完整研发与部署交付包 v{VERSION}

本目录同时提供普通用户运行部署、完整研发代码、原始标注数据、最终训练数据集、
训练实验结果和模型权重。为避免单文件超过 4 GB，研发数据按编号拆分；
这些文件共同组成一套完整交付。

## A. 只运行识别网页

使用 `00-mikania-portable-{PORTABLE_VERSION}.zip`。解压后按照包内快速开始文件或
零基础操作手册执行；不需要 Docker、预装 Python 或训练数据。

## B. 恢复完整研发项目

1. 新建一个空目录。
2. 将以下研发 ZIP 全部解压到同一个空目录，出现同名的
   `Agricultural_Species_Identification` 目录时选择合并：
   - `01-code-models-experiments.zip`
   - `02-labeled-raw-data.zip`
   - `03-training-ready-dataset.zip`
{auxiliary_names}
3. 进入合并后的 `Agricultural_Species_Identification` 目录。
4. 安装 Python 3.11，然后按照项目根目录 `README.md` 配置研发环境。

最终可直接训练的数据集位于：

`datasets/mikania_detection_v2_0_1/data.yaml`

原始已标注正样本位于 `raw-data`，辅助/待复核图片位于 `data`。
B0、B1、D1、S0 以及困难负样本实验的训练记录位于 `experiments`，最终部署模型位于
`model_store/mikania-yolo26s-v2`。D1 是最终采用模型；E1、E2 和双模型组合实验因误报率过高未用于部署。

## 完整性检查

- `SHA256SUMS.txt`：检查每个交付文件在复制后是否损坏；
- `FILE_MANIFEST.csv`：记录归档内每个研发文件的路径、大小和 SHA256；
- `CONTENTS.json`：记录各归档的文件数、容量和数据统计；
- 大文件建议使用 NTFS、exFAT、APFS 或 ext4 存储，不要使用单文件上限 4 GB 的 FAT32。

## 有意排除的重复/临时内容

- `.venv`、Python 缓存、测试缓存和训练缓存；
- 本地绝对路径形式的数据集符号链接；
- 原始数据已解压后留下的重复大 ZIP；
- 与薇甘菊项目无关的水下机器人专利 ZIP；
- 登录密码、SSH 私钥等认证信息。
"""


def _write_manifests(
    archive_records: list[ArchiveRecord],
    file_records: list[FileRecord],
    auxiliary_parts: int,
) -> None:
    summary_path = ROOT / "datasets" / "mikania_detection_v2_0_1" / "manifests" / "summary.json"
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_images = sum(
        1
        for path in (ROOT / "raw-data").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    raw_labels = sum(1 for path in (ROOT / "raw-data").iterdir() if path.suffix == ".txt")
    auxiliary_images = sum(
        1
        for path in (ROOT / "data").rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    contents = {
        "version": VERSION,
        "project": "Agricultural_Species_Identification",
        "archives": [asdict(record) for record in archive_records],
        "data_statistics": {
            "raw_image_count": raw_images,
            "raw_yolo_label_count": raw_labels,
            "auxiliary_image_count": auxiliary_images,
            "training_ready_image_count": dataset_summary["training_ready_image_count"],
            "training_ready_box_count": dataset_summary["training_ready_box_count"],
            "training_split_image_counts": dataset_summary["split_image_counts"],
        },
    }
    (HANDOFF_ROOT / "CONTENTS.json").write_text(
        json.dumps(contents, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (HANDOFF_ROOT / "FILE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["archive", "relative_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        for record in file_records:
            writer.writerow(asdict(record))
    (HANDOFF_ROOT / "README.md").write_text(
        _handoff_readme(auxiliary_parts),
        encoding="utf-8",
    )

    checksum_targets = sorted(
        path for path in HANDOFF_ROOT.iterdir() if path.name != "SHA256SUMS.txt"
    )
    checksum_lines = [f"{_sha256(path)}  {path.name}" for path in checksum_targets]
    (HANDOFF_ROOT / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )


def build() -> Path:
    portable = DIST_ROOT / f"mikania-portable-{PORTABLE_VERSION}.zip"
    if not portable.is_file():
        raise FileNotFoundError(f"Build the portable package first: {portable}")
    _safe_replace_handoff()

    portable_target = HANDOFF_ROOT / f"00-mikania-portable-{PORTABLE_VERSION}.zip"
    shutil.copy2(portable, portable_target)
    portable_record = ArchiveRecord(
        filename=portable_target.name,
        size_bytes=portable_target.stat().st_size,
        file_count=1,
        sha256=_sha256(portable_target),
    )

    archive_records = [portable_record]
    file_records: list[FileRecord] = []
    archive, files = _build_archive("01-code-models-experiments.zip", _code_entries())
    archive_records.append(archive)
    file_records.extend(files)

    archive, files = _build_archive("02-labeled-raw-data.zip", _regular_files(ROOT / "raw-data"))
    archive_records.append(archive)
    file_records.extend(files)

    training_dataset = ROOT / "datasets" / "mikania_detection_v2_0_1"
    archive, files = _build_archive(
        "03-training-ready-dataset.zip",
        _regular_files(training_dataset),
    )
    archive_records.append(archive)
    file_records.extend(files)

    auxiliary_parts = _partition(
        _regular_files(ROOT / "data"),
        MAX_AUXILIARY_PART_BYTES,
    )
    for index, entries in enumerate(auxiliary_parts, start=1):
        name = f"04-auxiliary-data-part{index:02d}.zip"
        archive, files = _build_archive(name, entries)
        archive_records.append(archive)
        file_records.extend(files)

    _write_manifests(archive_records, file_records, len(auxiliary_parts))
    return HANDOFF_ROOT


if __name__ == "__main__":
    destination = build()
    print(f"Complete handoff: {destination}")
