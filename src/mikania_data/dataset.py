from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .inventory import IMAGE_SUFFIXES, ImageRecord, scan_dataset

SplitName = Literal["train", "val", "test_internal", "test_external"]
InternalSplitName = Literal["train", "val", "test_internal"]
LinkMode = Literal["symlink", "copy"]


@dataclass(frozen=True, slots=True)
class DetectionDatasetConfig:
    version: str
    seed: int
    group_window_minutes: int
    near_duplicate_dhash_threshold: int
    split_fractions: dict[InternalSplitName, float]
    external_years: tuple[int, ...]
    class_names: tuple[str, ...]
    link_mode: LinkMode
    negative_review_cohorts: tuple[str, ...]
    missing_label_triage: dict[str, str]

    @classmethod
    def from_json(cls, path: Path) -> DetectionDatasetConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fractions = {
            name: float(payload["split_fractions"][name])
            for name in ("train", "val", "test_internal")
        }
        config = cls(
            version=str(payload["version"]),
            seed=int(payload["seed"]),
            group_window_minutes=int(payload["group_window_minutes"]),
            near_duplicate_dhash_threshold=int(
                payload["near_duplicate_dhash_threshold"]
            ),
            split_fractions=fractions,  # type: ignore[arg-type]
            external_years=tuple(int(year) for year in payload["external_years"]),
            class_names=tuple(str(name) for name in payload["class_names"]),
            link_mode=str(payload.get("link_mode", "symlink")),  # type: ignore[arg-type]
            negative_review_cohorts=tuple(
                str(name) for name in payload.get("negative_review_cohorts", [])
            ),
            missing_label_triage={
                str(name): str(status)
                for name, status in payload.get("missing_label_triage", {}).items()
            },
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.version.strip():
            raise ValueError("version cannot be empty")
        if self.group_window_minutes <= 0:
            raise ValueError("group_window_minutes must be positive")
        if not 0 <= self.near_duplicate_dhash_threshold <= 64:
            raise ValueError("near_duplicate_dhash_threshold must be between 0 and 64")
        if set(self.split_fractions) != {"train", "val", "test_internal"}:
            raise ValueError("split_fractions must define train, val, and test_internal")
        if any(value <= 0 for value in self.split_fractions.values()):
            raise ValueError("split fractions must be positive")
        if not math.isclose(sum(self.split_fractions.values()), 1.0, abs_tol=1e-9):
            raise ValueError("split fractions must sum to 1")
        if not self.external_years:
            raise ValueError("At least one external year is required")
        if not self.class_names:
            raise ValueError("At least one class name is required")
        if self.link_mode not in {"symlink", "copy"}:
            raise ValueError("link_mode must be 'symlink' or 'copy'")


@dataclass(frozen=True, slots=True)
class LabelStats:
    box_count: int
    area_sum: float


@dataclass(frozen=True, slots=True)
class DetectionSample:
    record: ImageRecord
    image_path: Path
    label_path: Path
    label_stats: LabelStats
    split_unit_id: str
    split: SplitName


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _canonical_label_stem(path: Path) -> str:
    return path.name[:-8] if path.name.endswith(".xml.txt") else path.stem


def _discover_label_paths(raw_root: Path) -> dict[str, Path]:
    labels: dict[str, Path] = {}
    for path in sorted(raw_root.glob("*.txt")):
        if path.name == "classes.txt":
            continue
        stem = _canonical_label_stem(path)
        if stem in labels:
            raise ValueError(
                f"Multiple labels resolve to the same image stem: {labels[stem]}, {path}"
            )
        labels[stem] = path
    return labels


def _parse_label(path: Path, class_count: int) -> LabelStats:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"Empty positive label file: {path}")
    area_sum = 0.0
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid field count in {path}:{line_number}")
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError as error:
            raise ValueError(f"Non-numeric label in {path}:{line_number}") from error
        if not 0 <= class_id < class_count:
            raise ValueError(f"Unknown class id in {path}:{line_number}: {class_id}")
        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            raise ValueError(f"Out-of-range box in {path}:{line_number}")
        if (
            x_center - width / 2 < -1e-6
            or x_center + width / 2 > 1 + 1e-6
            or y_center - height / 2 < -1e-6
            or y_center + height / 2 > 1 + 1e-6
        ):
            raise ValueError(f"Box crosses image boundary in {path}:{line_number}")
        area_sum += width * height
    return LabelStats(box_count=len(lines), area_sum=area_sum)


def _hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _sample_year(record: ImageRecord) -> int | None:
    if record.captured_at:
        return int(record.captured_at[:4])
    match = re.search(r"(?:19|20)\d{2}", record.relative_path)
    return int(match.group()) if match else None


def _stable_value(seed: int, value: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _build_split_units(
    records: list[ImageRecord], threshold: int
) -> dict[str, str]:
    disjoint = _DisjointSet(len(records))
    first_by_group: dict[str, int] = {}
    for index, record in enumerate(records):
        previous = first_by_group.setdefault(record.group_id, index)
        disjoint.union(previous, index)
    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
            if _hamming_distance(left.dhash, right.dhash) <= threshold:
                disjoint.union(left_index, right_index)

    paths_by_root: dict[int, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        paths_by_root[disjoint.find(index)].append(record.relative_path)

    unit_by_path: dict[str, str] = {}
    for paths in paths_by_root.values():
        payload = "\n".join(sorted(paths)).encode()
        unit_id = f"unit-{hashlib.sha256(payload).hexdigest()[:12]}"
        for path in paths:
            unit_by_path[path] = unit_id
    return unit_by_path


def _allocate_internal_units(
    unit_samples: dict[str, list[tuple[ImageRecord, LabelStats]]],
    fractions: dict[InternalSplitName, float],
    seed: int,
) -> dict[str, InternalSplitName]:
    split_names: tuple[InternalSplitName, ...] = ("train", "val", "test_internal")
    total_images = sum(len(samples) for samples in unit_samples.values())
    total_boxes = sum(
        stats.box_count for samples in unit_samples.values() for _, stats in samples
    )
    targets = {
        name: {
            "images": max(1.0, total_images * fractions[name]),
            "boxes": max(1.0, total_boxes * fractions[name]),
        }
        for name in split_names
    }
    current = {
        name: {"images": 0, "boxes": 0}
        for name in split_names
    }
    allocation: dict[str, InternalSplitName] = {}
    ordered_units = sorted(
        unit_samples,
        key=lambda unit: (
            -len(unit_samples[unit]),
            -sum(stats.box_count for _, stats in unit_samples[unit]),
            _stable_value(seed, unit),
        ),
    )
    for unit in ordered_units:
        image_count = len(unit_samples[unit])
        box_count = sum(stats.box_count for _, stats in unit_samples[unit])
        choice = min(
            split_names,
            key=lambda name: (
                (current[name]["images"] + image_count) / targets[name]["images"]
                + (current[name]["boxes"] + box_count) / targets[name]["boxes"],
                _stable_value(seed, f"{unit}:{name}"),
            ),
        )
        allocation[unit] = choice
        current[choice]["images"] += image_count
        current[choice]["boxes"] += box_count
    return allocation


def _relative_source(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _link_or_copy(source: Path, target: Path, mode: LinkMode) -> None:
    if mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def _write_dataset_yaml(output: Path, *, external: bool) -> None:
    test_directory = "images/test_external" if external else "images/test_internal"
    content = (
        "train: images/train\n"
        "val: images/val\n"
        f"test: {test_directory}\n\n"
        "names:\n"
        "  0: mikania\n"
    )
    name = "data_external.yaml" if external else "data.yaml"
    (output / name).write_text(content, encoding="utf-8")


def _negative_review_paths(
    negative_root: Path, cohorts: tuple[str, ...]
) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    for cohort in cohorts:
        cohort_dir = negative_root / cohort
        if not cohort_dir.is_dir():
            continue
        output.extend(
            (cohort, path)
            for path in sorted(cohort_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return output


def build_detection_dataset(
    raw_root: Path,
    negative_root: Path,
    output_dir: Path,
    config: DetectionDatasetConfig,
) -> Path:
    config.validate()
    raw = raw_root.expanduser().resolve()
    negatives = negative_root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    project_root = Path.cwd().resolve()
    if not raw.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw}")
    if not negatives.is_dir():
        raise FileNotFoundError(f"Negative candidate directory does not exist: {negatives}")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new versioned path: {output}")
    if output == raw or raw in output.parents:
        raise ValueError("Output cannot be placed inside raw-data")

    records, bad_images = scan_dataset(raw, config.group_window_minutes)
    if bad_images:
        raise ValueError(f"Raw dataset contains {len(bad_images)} unreadable image(s)")
    labels_by_stem = _discover_label_paths(raw)
    record_by_stem = {Path(record.relative_path).stem: record for record in records}
    orphan_labels = sorted(set(labels_by_stem) - set(record_by_stem))
    if orphan_labels:
        raise ValueError(f"Orphan label stems: {orphan_labels}")

    labeled_records = [
        record for record in records if Path(record.relative_path).stem in labels_by_stem
    ]
    missing_records = [
        record for record in records if Path(record.relative_path).stem not in labels_by_stem
    ]
    label_stats_by_path = {
        record.relative_path: _parse_label(
            labels_by_stem[Path(record.relative_path).stem], len(config.class_names)
        )
        for record in labeled_records
    }
    unit_by_path = _build_split_units(
        labeled_records, config.near_duplicate_dhash_threshold
    )

    samples_by_unit: dict[str, list[tuple[ImageRecord, LabelStats]]] = defaultdict(list)
    external_units: set[str] = set()
    internal_units: set[str] = set()
    for record in labeled_records:
        unit = unit_by_path[record.relative_path]
        stats = label_stats_by_path[record.relative_path]
        samples_by_unit[unit].append((record, stats))
        if _sample_year(record) in config.external_years:
            external_units.add(unit)
        else:
            internal_units.add(unit)
    mixed_units = external_units & internal_units
    if mixed_units:
        raise ValueError(
            "Near-duplicate units cross the temporal external-test boundary: "
            f"{sorted(mixed_units)}"
        )
    internal_samples = {
        unit: samples_by_unit[unit] for unit in sorted(internal_units)
    }
    internal_allocation = _allocate_internal_units(
        internal_samples, config.split_fractions, config.seed
    )

    samples: list[DetectionSample] = []
    for record in labeled_records:
        unit = unit_by_path[record.relative_path]
        split: SplitName = (
            "test_external" if unit in external_units else internal_allocation[unit]
        )
        image_path = raw / record.relative_path
        label_path = labels_by_stem[Path(record.relative_path).stem]
        samples.append(
            DetectionSample(
                record=record,
                image_path=image_path,
                label_path=label_path,
                label_stats=label_stats_by_path[record.relative_path],
                split_unit_id=unit,
                split=split,
            )
        )

    negative_review = _negative_review_paths(
        negatives, config.negative_review_cohorts
    )
    output.mkdir(parents=True)
    for split in ("train", "val", "test_internal", "test_external"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    (output / "manifests").mkdir()
    (output / "pending" / "missing_labels" / "images").mkdir(parents=True)
    (output / "pending" / "negative_review" / "images").mkdir(parents=True)

    manifest_path = output / "manifests" / "samples.csv"
    manifest_fields = [
        "image",
        "label",
        "split",
        "source_image",
        "source_label",
        "source_label_naming",
        "group_id",
        "split_unit_id",
        "captured_at",
        "camera",
        "width",
        "height",
        "box_count",
        "box_area_sum",
        "sha256",
        "dhash",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for sample in sorted(samples, key=lambda item: (item.split, item.image_path.name)):
            relative_image = Path("images") / sample.split / sample.image_path.name
            relative_label = Path("labels") / sample.split / f"{sample.image_path.stem}.txt"
            _link_or_copy(
                sample.image_path, output / relative_image, config.link_mode
            )
            shutil.copy2(sample.label_path, output / relative_label)
            writer.writerow(
                {
                    "image": relative_image.as_posix(),
                    "label": relative_label.as_posix(),
                    "split": sample.split,
                    "source_image": _relative_source(sample.image_path, project_root),
                    "source_label": _relative_source(sample.label_path, project_root),
                    "source_label_naming": (
                        "xml_txt"
                        if sample.label_path.name.endswith(".xml.txt")
                        else "plain_txt"
                    ),
                    "group_id": sample.record.group_id,
                    "split_unit_id": sample.split_unit_id,
                    "captured_at": sample.record.captured_at,
                    "camera": sample.record.camera,
                    "width": sample.record.width,
                    "height": sample.record.height,
                    "box_count": sample.label_stats.box_count,
                    "box_area_sum": f"{sample.label_stats.area_sum:.9f}",
                    "sha256": sample.record.sha256,
                    "dhash": sample.record.dhash,
                }
            )

    missing_manifest = output / "manifests" / "pending_missing_labels.csv"
    with missing_manifest.open("w", encoding="utf-8", newline="") as handle:
        fields = ["image", "source_image", "group_id", "triage", "review_status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(missing_records, key=lambda item: item.relative_path):
            source = raw / record.relative_path
            target_name = Path(record.relative_path).name
            relative_image = Path("pending/missing_labels/images") / target_name
            _link_or_copy(source, output / relative_image, config.link_mode)
            writer.writerow(
                {
                    "image": relative_image.as_posix(),
                    "source_image": _relative_source(source, project_root),
                    "group_id": record.group_id,
                    "triage": config.missing_label_triage.get(
                        target_name, "expert_review"
                    ),
                    "review_status": "pending",
                }
            )

    negative_manifest = output / "manifests" / "pending_negative_review.csv"
    with negative_manifest.open("w", encoding="utf-8", newline="") as handle:
        fields = ["image", "source_image", "candidate_cohort", "review_status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (cohort, source) in enumerate(negative_review, start=1):
            target_name = f"N{index:03d}__{source.name}"
            relative_image = Path("pending/negative_review/images") / target_name
            _link_or_copy(source, output / relative_image, config.link_mode)
            writer.writerow(
                {
                    "image": relative_image.as_posix(),
                    "source_image": _relative_source(source, project_root),
                    "candidate_cohort": cohort,
                    "review_status": "pending_expert_review",
                }
            )

    _write_dataset_yaml(output, external=False)
    _write_dataset_yaml(output, external=True)
    split_counts = {
        split: sum(sample.split == split for sample in samples)
        for split in ("train", "val", "test_internal", "test_external")
    }
    split_box_counts = {
        split: sum(
            sample.label_stats.box_count for sample in samples if sample.split == split
        )
        for split in ("train", "val", "test_internal", "test_external")
    }
    summary = {
        "version": config.version,
        "seed": config.seed,
        "class_names": list(config.class_names),
        "source_image_count": len(records),
        "training_ready_image_count": len(samples),
        "training_ready_box_count": sum(
            sample.label_stats.box_count for sample in samples
        ),
        "split_image_counts": split_counts,
        "split_box_counts": split_box_counts,
        "split_unit_counts": {
            split: len(
                {sample.split_unit_id for sample in samples if sample.split == split}
            )
            for split in split_counts
        },
        "missing_label_count": len(missing_records),
        "negative_review_candidate_count": len(negative_review),
        "canonicalized_xml_txt_label_count": sum(
            sample.label_path.name.endswith(".xml.txt") for sample in samples
        ),
        "near_duplicate_dhash_threshold": config.near_duplicate_dhash_threshold,
        "external_years": list(config.external_years),
        "link_mode": config.link_mode,
    }
    summary["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (output / "manifests" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Mikania Detection Dataset v1\n\n"
        "This directory is generated from `raw-data` without modifying source files.\n\n"
        "- `data.yaml`: grouped train/val/internal-test protocol.\n"
        "- `data_external.yaml`: same train/val data with the 2026 temporal test set.\n"
        "- `manifests/samples.csv`: full source and split provenance.\n"
        "- `pending/missing_labels`: excluded images requiring annotation.\n"
        "- `pending/negative_review`: similar-species/background candidates requiring "
        "expert confirmation before empty-label inclusion.\n\n"
        "Image files are local symlinks by default; label files are canonical `.txt` copies.\n",
        encoding="utf-8",
    )
    return output
