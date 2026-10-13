from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .inventory import ROOT_COHORT, UAV_COHORT, BadImage, ImageRecord


@dataclass(frozen=True, slots=True)
class PilotConfig:
    seed: int
    group_window_minutes: int
    ground_quotas: dict[str, int]
    uav_count: int
    link_mode: Literal["symlink", "copy"] = "symlink"

    @classmethod
    def from_json(cls, path: Path) -> PilotConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = cls(
            seed=int(payload["seed"]),
            group_window_minutes=int(payload["group_window_minutes"]),
            ground_quotas={
                str(cohort): int(count) for cohort, count in payload["ground_quotas"].items()
            },
            uav_count=int(payload["uav_count"]),
            link_mode=str(payload.get("link_mode", "symlink")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.group_window_minutes <= 0:
            raise ValueError("group_window_minutes must be positive")
        if self.uav_count < 0 or any(count < 0 for count in self.ground_quotas.values()):
            raise ValueError("Pilot quotas cannot be negative")
        if self.link_mode not in {"symlink", "copy"}:
            raise ValueError("link_mode must be 'symlink' or 'copy'")


@dataclass(frozen=True, slots=True)
class SelectedImage:
    pilot_id: str
    record: ImageRecord


def _stable_tiebreak(record: ImageRecord, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{record.relative_path}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _squared_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def diverse_group_sample(records: list[ImageRecord], count: int, seed: int) -> list[ImageRecord]:
    if count > len(records):
        raise ValueError(f"Requested {count} images, but only {len(records)} are available")
    if count == 0:
        return []

    remaining = list(records)
    selected: list[ImageRecord] = []
    used_groups: set[str] = set()
    while len(selected) < count:
        unseen_group_candidates = [
            record for record in remaining if record.group_id not in used_groups
        ]
        candidates = unseen_group_candidates or remaining
        if not selected:
            choice = min(candidates, key=lambda item: _stable_tiebreak(item, seed))
        else:
            choice = max(
                candidates,
                key=lambda item: (
                    min(
                        _squared_distance(item.feature, chosen.feature) for chosen in selected
                    ),
                    -_stable_tiebreak(item, seed),
                ),
            )
        selected.append(choice)
        used_groups.add(choice.group_id)
        remaining.remove(choice)
    return selected


def select_pilot(records: list[ImageRecord], config: PilotConfig) -> list[SelectedImage]:
    config.validate()
    selected_ground: list[ImageRecord] = []
    for cohort, quota in config.ground_quotas.items():
        cohort_records = [
            record
            for record in records
            if record.modality == "ground" and record.cohort == cohort
        ]
        selected_ground.extend(diverse_group_sample(cohort_records, quota, config.seed))

    uav_records = [
        record
        for record in records
        if record.modality == "uav" and record.cohort == UAV_COHORT
    ]
    selected_uav = diverse_group_sample(uav_records, config.uav_count, config.seed + 1)

    ground_sorted = sorted(
        selected_ground,
        key=lambda item: (
            item.cohort != ROOT_COHORT,
            item.cohort,
            item.relative_path,
        ),
    )
    uav_sorted = sorted(selected_uav, key=lambda item: item.relative_path)
    output = [
        SelectedImage(pilot_id=f"G{index:03d}", record=record)
        for index, record in enumerate(ground_sorted, start=1)
    ]
    output.extend(
        SelectedImage(pilot_id=f"U{index:03d}", record=record)
        for index, record in enumerate(uav_sorted, start=1)
    )
    return output


def materialize_pilot(
    selected: list[SelectedImage],
    data_root: Path,
    output_dir: Path,
    link_mode: str,
    excluded_bad_images: Sequence[BadImage] = (),
) -> Path:
    source_root = data_root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new versioned path: {output}")
    if output == source_root or source_root in output.parents:
        raise ValueError("Pilot output cannot be placed inside the source data directory")

    for modality in ("ground", "uav"):
        (output / "images" / modality).mkdir(parents=True, exist_ok=True)
        (output / "labels" / modality).mkdir(parents=True, exist_ok=True)

    manifest_path = output / "manifest.csv"
    fieldnames = [
        "pilot_id",
        "modality",
        "cohort",
        "candidate_label",
        "source_relative_path",
        "pilot_image",
        "expected_label",
        "annotation_type",
        "group_id",
        "captured_at",
        "camera",
        "review_status",
        "expert_note",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in selected:
            record = item.record
            source = (source_root / record.relative_path).resolve()
            if source_root not in source.parents:
                raise ValueError(f"Source path escaped data root: {record.relative_path}")
            target_name = f"{item.pilot_id}__{source.name}"
            relative_image = Path("images") / record.modality / target_name
            target = output / relative_image
            if link_mode == "symlink":
                target.symlink_to(source)
            elif link_mode == "copy":
                shutil.copy2(source, target)
            else:
                raise ValueError("link_mode must be 'symlink' or 'copy'")

            label_suffix = ".txt" if record.modality == "ground" else ".png"
            relative_label = (
                Path("labels") / record.modality / f"{Path(target_name).stem}{label_suffix}"
            )
            writer.writerow(
                {
                    "pilot_id": item.pilot_id,
                    "modality": record.modality,
                    "cohort": record.cohort,
                    "candidate_label": record.candidate_label,
                    "source_relative_path": record.relative_path,
                    "pilot_image": relative_image.as_posix(),
                    "expected_label": relative_label.as_posix(),
                    "annotation_type": (
                        "bbox_yolo" if record.modality == "ground" else "semantic_mask_png"
                    ),
                    "group_id": record.group_id,
                    "captured_at": record.captured_at,
                    "camera": record.camera,
                    "review_status": "pending",
                    "expert_note": "",
                }
            )

    if excluded_bad_images:
        with (output / "excluded_bad_images.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path", "error"])
            writer.writeheader()
            for item in excluded_bad_images:
                writer.writerow(
                    {"relative_path": item.relative_path, "error": item.error}
                )

    summary = {
        "total": len(selected),
        "ground": sum(item.record.modality == "ground" for item in selected),
        "uav": sum(item.record.modality == "uav" for item in selected),
        "cohorts": dict(
            sorted(
                {
                    cohort: sum(item.record.cohort == cohort for item in selected)
                    for cohort in {item.record.cohort for item in selected}
                }.items()
            )
        ),
        "link_mode": link_mode,
        "excluded_bad_image_count": len(excluded_bad_images),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output
