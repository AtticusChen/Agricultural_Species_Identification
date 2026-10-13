#!/usr/bin/env python3
"""Build a versioned detection dataset with reviewed hard negatives.

The v1 dataset and raw source files are never modified. Existing split
assignments are preserved, while reviewed negative cohorts are allocated in
capture-sequence groups to avoid near-duplicate leakage across splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

SPLITS = ("train", "val", "test_internal", "test_external")
INTERNAL_SPLITS = ("train", "val", "test_internal")
FRACTIONS = {"train": 0.70, "val": 0.15, "test_internal": 0.15}
JPEG_SUFFIXES = {".jpg", ".jpeg"}
JPEG_EOI = b"\xff\xd9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--negative-manifest", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="mikania_detection_v2.0.0")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--copy", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_value(seed: int, value: str) -> int:
    payload = f"{seed}:{value}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def relative_source(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def link_or_copy(source: Path, target: Path, *, copy: bool) -> None:
    if copy:
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError:
        target.symlink_to(source.resolve())


def ensure_jpeg_eoi(path: Path) -> bool:
    """Append the missing JPEG end marker without re-encoding pixels."""
    if path.suffix.lower() not in JPEG_SUFFIXES or path.stat().st_size < 2:
        return False
    with path.open("rb") as handle:
        handle.seek(-2, 2)
        if handle.read() == JPEG_EOI:
            return False
    with path.open("ab") as handle:
        handle.write(JPEG_EOI)
    return True


def timestamp_from_name(path: Path) -> datetime | None:
    match = re.search(r"((?:19|20)\d{6})[^0-9]?([0-2]\d[0-5]\d[0-5]\d)", path.stem)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def negative_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_cohort: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_cohort[row["candidate_cohort"]].append(row)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for cohort, cohort_rows in sorted(by_cohort.items()):
        dated = sorted(
            cohort_rows,
            key=lambda row: (
                timestamp_from_name(Path(row["source_image"])) or datetime.max,
                row["source_image"],
            ),
        )
        sequence_index = 0
        previous: datetime | None = None
        for row in dated:
            captured_at = timestamp_from_name(Path(row["source_image"]))
            if previous is None or captured_at is None or (captured_at - previous).total_seconds() > 300:
                sequence_index += 1
            group_key = f"{cohort}|sequence-{sequence_index:03d}"
            groups[group_key].append(row)
            previous = captured_at
    return groups


def allocate_groups(
    groups: dict[str, list[dict[str, str]]], seed: int
) -> dict[str, str]:
    total = sum(map(len, groups.values()))
    targets = {split: total * FRACTIONS[split] for split in INTERNAL_SPLITS}
    current = Counter({split: 0 for split in INTERNAL_SPLITS})
    allocation: dict[str, str] = {}
    ordered = sorted(groups, key=lambda key: (-len(groups[key]), stable_value(seed, key)))
    for group_key in ordered:
        size = len(groups[group_key])
        split = min(
            INTERNAL_SPLITS,
            key=lambda name: (
                (current[name] + size) / max(targets[name], 1),
                stable_value(seed, f"{group_key}:{name}"),
            ),
        )
        allocation[group_key] = split
        current[split] += size
    return allocation


def image_facts(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        width, height = image.size
        tiny = image.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(tiny.getdata())
    bits = []
    for row in range(8):
        offset = row * 9
        bits.extend(pixels[offset + column] > pixels[offset + column + 1] for column in range(8))
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return width, height, f"{value:016x}"


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_yaml(output: Path, external: bool) -> None:
    test = "test_external" if external else "test_internal"
    name = "data_external.yaml" if external else "data.yaml"
    (output / name).write_text(
        "train: images/train\n"
        "val: images/val\n"
        f"test: images/{test}\n\n"
        "names:\n"
        "  0: mikania\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    source = args.source_dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    source_fields, source_rows = load_csv(source / "manifests" / "samples.csv")
    _, candidate_rows = load_csv(args.negative_manifest.resolve())
    _, correction_rows = load_csv(args.corrections.resolve())
    corrections = {row["source_image"]: row for row in correction_rows}
    unknown_corrections = sorted(set(corrections) - {row["source_image"] for row in source_rows})
    if unknown_corrections:
        raise ValueError(f"Corrections not found in v1 manifest: {unknown_corrections}")

    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "manifests").mkdir(parents=True)

    added_fields = ["sample_role", "annotation_action", "review_note"]
    manifest_fields = source_fields + [field for field in added_fields if field not in source_fields]
    output_rows: list[dict[str, str]] = []
    repaired_images: list[str] = []

    for row in source_rows:
        new_row = dict(row)
        relative_image = Path(row["image"])
        relative_label = Path(row["label"])
        source_image = source / relative_image
        source_label = source / relative_label
        target_image = output / relative_image
        target_label = output / relative_label
        link_or_copy(source_image, target_image, copy=args.copy)
        source_repaired = ensure_jpeg_eoi(target_image)
        if source_repaired:
            repaired_images.append(relative_image.as_posix())
            new_row["parent_sha256"] = row.get("sha256", "")
            new_row["sha256"] = sha256(target_image)
            new_row["jpeg_eoi_appended"] = "1"

        correction = corrections.get(row["source_image"])
        if correction:
            target_label.write_text("", encoding="utf-8")
            new_row.update(
                {
                    "box_count": "0",
                    "box_area_sum": "0.000000000",
                    "sample_role": "hard_negative",
                    "annotation_action": correction["action"],
                    "review_note": correction["reason"],
                }
            )
        else:
            shutil.copy2(source_label, target_label)
            new_row.update(
                {
                    "sample_role": "positive",
                    "annotation_action": "preserved_from_v1",
                    "review_note": "",
                }
            )
        output_rows.append(new_row)

    groups = negative_groups(candidate_rows)
    allocation = allocate_groups(groups, args.seed)
    group_by_source = {
        row["source_image"]: group_key
        for group_key, rows in groups.items()
        for row in rows
    }
    for index, row in enumerate(sorted(candidate_rows, key=lambda item: item["source_image"]), start=1):
        original = project_root / row["source_image"]
        if not original.is_file():
            raise FileNotFoundError(f"Reviewed negative image not found: {original}")
        group_key = group_by_source[row["source_image"]]
        split = allocation[group_key]
        suffix = original.suffix.lower() if original.suffix else ".jpg"
        target_name = f"NEG{index:03d}_{original.stem}{suffix}"
        relative_image = Path("images") / split / target_name
        relative_label = Path("labels") / split / f"NEG{index:03d}_{original.stem}.txt"
        link_or_copy(original, output / relative_image, copy=args.copy)
        negative_repaired = ensure_jpeg_eoi(output / relative_image)
        if negative_repaired:
            repaired_images.append(relative_image.as_posix())
        (output / relative_label).write_text("", encoding="utf-8")
        width, height, dhash = image_facts(original)
        unit_hash = hashlib.sha256(group_key.encode()).hexdigest()[:12]
        base = {field: "" for field in manifest_fields}
        base.update(
            {
                "image": relative_image.as_posix(),
                "label": relative_label.as_posix(),
                "split": split,
                "source_image": relative_source(original, project_root),
                "source_label": "",
                "source_label_naming": "empty_negative_label",
                "group_id": group_key,
                "split_unit_id": f"negative-unit-{unit_hash}",
                "captured_at": (timestamp_from_name(original) or "").isoformat(sep=" ") if timestamp_from_name(original) else "",
                "camera": "",
                "width": str(width),
                "height": str(height),
                "box_count": "0",
                "box_area_sum": "0.000000000",
                "parent_sha256": "",
                "sha256": sha256(output / relative_image),
                "dhash": dhash,
                "jpeg_eoi_appended": "1" if negative_repaired else "0",
                "sample_role": "reviewed_negative",
                "annotation_action": "promoted_from_negative_review",
                "review_note": f'候选目录“{row["candidate_cohort"]}”，经整批可视化核验后作为非薇甘菊样本',
            }
        )
        output_rows.append(base)

    manifest_path = output / "manifests" / "samples.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(sorted(output_rows, key=lambda row: (row["split"], row["image"])))

    shutil.copy2(args.corrections, output / "manifests" / "dataset_corrections.csv")
    shutil.copy2(args.negative_manifest, output / "manifests" / "reviewed_negative_sources.csv")
    write_yaml(output, external=False)
    write_yaml(output, external=True)

    split_counts = Counter(row["split"] for row in output_rows)
    split_boxes = Counter()
    role_counts = Counter(row["sample_role"] for row in output_rows)
    negative_split_counts = Counter(
        row["split"] for row in output_rows if row["sample_role"] != "positive"
    )
    source_summary_path = source / "manifests" / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    for row in output_rows:
        split_boxes[row["split"]] += int(row["box_count"])
    summary = {
        "version": args.version,
        "seed": args.seed,
        "parent_version": source_summary.get("version"),
        "parent_summary_sha256": sha256(source_summary_path),
        "class_names": ["mikania"],
        "training_ready_image_count": len(output_rows),
        "training_ready_box_count": sum(split_boxes.values()),
        "split_image_counts": {split: split_counts[split] for split in SPLITS},
        "split_box_counts": {split: split_boxes[split] for split in SPLITS},
        "sample_role_counts": dict(sorted(role_counts.items())),
        "negative_split_counts": {split: negative_split_counts[split] for split in SPLITS},
        "reviewed_negative_candidate_count": len(candidate_rows),
        "converted_hard_negative_count": len(correction_rows),
        "manifest_sha256": sha256(manifest_path),
        "storage_mode": "copy" if args.copy else "hardlink_or_symlink",
        "jpeg_eoi_repair_method": "append_ffd9_without_reencoding",
        "jpeg_eoi_repaired_count": len(repaired_images),
    }
    (output / "manifests" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Mikania Detection Dataset v2\n\n"
        "在不修改 v1 与原始数据的前提下生成：\n\n"
        "- 保留 v1 的正样本与分组划分；\n"
        "- 将 8 张错误正标照片改为空标签硬负样本；\n"
        "- 加入 123 张相似植物/背景负样本；\n"
        "- 负样本按物种与五分钟拍摄序列分组划分，避免相邻照片跨集合泄漏。\n",
        encoding="utf-8",
    )
    (output / "manifests" / "jpeg_eoi_repairs.csv").write_text(
        "image,action\n"
        + "".join(f'{image},append_ffd9\n' for image in repaired_images),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
