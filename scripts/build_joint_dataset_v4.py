#!/usr/bin/env python3
"""Create a traceable, non-destructive joint scene/region classification dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing empty manifest: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def square(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ImageOps.pad(image, (768, 768), color=(114, 114, 114)).save(path, quality=95)


def load_image(path: Path) -> tuple[Image.Image, str]:
    """Fully decode a source; only a missing JPEG EOI may be repaired in memory."""
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            return image, ""
    except OSError:
        data = path.read_bytes()
        if not data.startswith(b"\xff\xd8") or data.endswith(b"\xff\xd9"):
            raise
        with Image.open(io.BytesIO(data + b"\xff\xd9")) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
        return image, "jpeg_eoi_appended_in_memory"


def audit_source_splits(samples: list[dict], reviews: list[dict], upload: Path) -> None:
    """Audit original sources, including uncertain images supplying reviewed crops."""
    group_splits: dict[str, set[str]] = {}
    hash_splits: dict[str, set[str]] = {}
    pixel_splits: dict[str, set[str]] = {}
    integrity: dict[Path, dict[str, str]] = {}

    def include(path: Path, domain: str, group_id: str, split: str) -> dict[str, str]:
        if not group_id.strip():
            raise ValueError("Every source image requires a nonempty group_id")
        path = path.resolve()
        if path not in integrity:
            image, warning = load_image(path)
            pixel_hash = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode())
            pixel_hash.update(image.tobytes())
            integrity[path] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pixel_sha256": pixel_hash.hexdigest(),
                "decode_warning": warning,
            }
        record = integrity[path]
        group_splits.setdefault(f"{domain}:{group_id}", set()).add(split)
        hash_splits.setdefault(record["sha256"], set()).add(split)
        pixel_splits.setdefault(record["pixel_sha256"], set()).add(split)
        return record

    for row in samples:
        row.update(include(Path(row["image"]), row["domain"], row["group_id"], row["split"]))
    for review in reviews:
        # Even an uncertain whole-image label may supply an explicitly reviewed
        # negative crop later. Audit every review source before creating any crop.
        record = include(
            upload / "airport" / review["filename"],
            "uav",
            review["group_id"],
            review["split"],
        )
        review.update({f"source_{key}": value for key, value in record.items()})
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("Group split leakage")
    if any(len(splits) != 1 for splits in hash_splits.values()):
        raise ValueError("Exact image leakage")
    if any(len(splits) != 1 for splits in pixel_splits.values()):
        raise ValueError("Decoded pixel image leakage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground", type=Path, required=True)
    parser.add_argument("--upload", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-repeat", type=int, default=2)
    parser.add_argument("--version", default="mikania_joint_v4_0_0")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    ground = args.ground.resolve()
    samples = []
    for row in read_csv(ground / "manifests/samples.csv"):
        samples.append(
            dict(
                image=str(ground / row["image"]),
                split=row["split"],
                domain="ground",
                label="positive" if row["sample_role"] == "positive" else "negative",
                group_id=row["split_unit_id"],
                provenance="legacy_v2_review",
                original_review_label="",
                source=row["source_image"],
                regions="",
                ground_label=str(ground / row["label"]),
            )
        )
    reviews = read_csv(args.reviews)
    seen = set()
    for row in reviews:
        if row["filename"] in seen:
            raise ValueError("Duplicate review: " + row["filename"])
        seen.add(row["filename"])
        if row["label"] not in {"positive", "negative", "uncertain"}:
            raise ValueError("Unknown label")
        if row["split"] not in {"train", "val", "test_uav"}:
            raise ValueError("Unknown UAV split")
        if row["label"] == "uncertain":
            continue
        samples.append(
            dict(
                image=str(args.upload.resolve() / "airport" / row["filename"]),
                split=row["split"],
                domain="uav",
                label=row["label"],
                group_id=row["group_id"],
                provenance=row.get("provenance") or "assistant_visual_provisional",
                original_review_label=row.get("original_review_label", row["label"]),
                source="airport/" + row["filename"],
                regions=row.get("regions", ""),
                ground_label="",
            )
        )
    # Every group and every exact image is assigned to one split only.
    audit_source_splits(samples, reviews, args.upload.resolve())
    region_rows = []
    for i, row in enumerate(samples):
        image_path = Path(row["image"])
        row["id"] = f"{row['domain']}_{i:04d}_{image_path.stem}"
    # Ensure class directories exist, even when one domain has no members of a class.
    for view in ("scene", "patch"):
        for split in ("train", "val"):
            for cl in ("mikania", "non_mikania"):
                (out / view / split / cl).mkdir(parents=True, exist_ok=True)
    for row in samples:
        if row["split"] not in {"train", "val"}:
            row["scene_file"] = ""
            continue
        image, decode_warning = load_image(Path(row["image"]))
        if decode_warning != row["decode_warning"]:
            raise ValueError("Image decoding changed since source audit")
        cl = "mikania" if row["label"] == "positive" else "non_mikania"
        relative = f"scene/{row['split']}/{cl}/{row['id']}.jpg"
        square(image, out / relative)
        row["scene_file"] = relative
        if row["split"] == "train" and row["label"] == "negative":
            for j in range(1, args.negative_repeat):
                extra = out / f"scene/train/{cl}/{row['id']}_repeat{j}.jpg"
                extra.symlink_to(Path(relative).name)
        regions = []
        if row["label"] == "negative":
            # Only whole-image negative references are allowed to supply all negative crops.
            side = min(1280, image.width, image.height)
            for x, y in (
                (0, 0),
                (image.width - side, 0),
                (0, image.height - side),
                (image.width - side, image.height - side),
            ):
                regions.append((x, y, x + side, y + side))
        elif row["domain"] == "ground":
            for line in Path(row["ground_label"]).read_text().splitlines():
                values = line.split()
                if len(values) != 5 or values[0] != "0":
                    raise ValueError("Unexpected legacy box format")
                cx, cy, w, h = map(float, values[1:])
                side = max(w * image.width, h * image.height) * 1.4
                x, y = cx * image.width, cy * image.height
                regions.append(
                    (
                        max(0, x - side / 2),
                        max(0, y - side / 2),
                        min(image.width, x + side / 2),
                        min(image.height, y + side / 2),
                    )
                )
        else:
            # UAV positives need explicitly reviewed regions; no automatic positive tiling.
            for region in row["regions"].split(";"):
                if not region.strip():
                    continue
                x1, y1, x2, y2 = map(float, region.split())
                if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                    raise ValueError("Invalid reviewed normalized region")
                regions.append(
                    (x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height)
                )
        for j, box in enumerate(regions):
            box = tuple(round(v) for v in box)
            relative = f"patch/{row['split']}/{cl}/{row['id']}_{j:02d}.jpg"
            square(image.crop(box), out / relative)
            region_rows.append(
                dict(
                    image=relative,
                    source_image=row["image"],
                    split=row["split"],
                    domain=row["domain"],
                    label=row["label"],
                    group_id=row["group_id"],
                    box_xyxy=" ".join(map(str, box)),
                    provenance=(
                        "assistant_visual_provisional"
                        if row["domain"] == "uav"
                        else row["provenance"]
                    ),
                    source_label_provenance=row["provenance"],
                    source_sha256=row["sha256"],
                    source_pixel_sha256=row["pixel_sha256"],
                    source_decode_warning=row["decode_warning"],
                )
            )
    negative_crops = []
    for review in reviews:
        if review["split"] not in {"train", "val"}:
            continue
        selected = review.get("negative_regions", "")
        if not selected:
            continue
        source = args.upload.resolve() / "airport" / review["filename"]
        image, decode_warning = load_image(source)
        if decode_warning != review["source_decode_warning"]:
            raise ValueError("Image decoding changed since source audit")
        for index, region in enumerate(selected.split(";")):
            x1, y1, x2, y2 = map(float, region.split())
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("Invalid reviewed negative region")
            box = tuple(
                round(v)
                for v in (x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height)
            )
            name = f"uav_negative_{source.stem}_{index:02d}.jpg"
            relative = f"patch/{review['split']}/non_mikania/{name}"
            square(image.crop(box), out / relative)
            record = dict(
                image=relative,
                source_image=str(source),
                split=review["split"],
                domain="uav",
                label="negative",
                group_id=review["group_id"],
                box_xyxy=" ".join(map(str, box)),
                provenance="assistant_visual_provisional",
                source_label_provenance=(
                    review.get("provenance") or "assistant_visual_provisional"
                ),
                source_sha256=review["source_sha256"],
                source_pixel_sha256=review["source_pixel_sha256"],
                source_decode_warning=review["source_decode_warning"],
            )
            region_rows.append(record)
            negative_crops.append(record)
            # A reviewed crop is a valid separate classifier training input. It is
            # never counted as a negative original photo or as an independent site.
            square(image.crop(box), out / f"scene/{review['split']}/non_mikania/{name}")
    write_csv(out / "manifests/samples.csv", samples)
    write_csv(out / "manifests/regions.csv", region_rows)
    write_csv(out / "manifests/uav_reviews.csv", reviews)
    summary = dict(
        version=args.version,
        source=str(args.upload.resolve()),
        ground_parent=str(ground),
        negative_repeat=args.negative_repeat,
        scene_preprocess="PIL ImageOps.pad 768x768 fill (114,114,114)",
        region_preprocess="same pad 768x768; only reviewed UAV positive regions",
        counts={
            "|".join(k): v
            for k, v in Counter((r["domain"], r["split"], r["label"]) for r in samples).items()
        },
        region_counts={
            "|".join(k): v
            for k, v in Counter((r["domain"], r["split"], r["label"]) for r in region_rows).items()
        },
        uav_uncertain=sum(r["label"] == "uncertain" for r in reviews),
        uav_negative_region_inputs=len(negative_crops),
        label_provenance_counts=dict(Counter(r["provenance"] for r in samples)),
        label_warning=(
            "UAV photograph labels retain supplied provenance; user-reported labels are not "
            "botanist-confirmed truth. UAV ROI labels remain assistant visual provisional."
        ),
        decode_policy="Strict full decode; missing JPEG EOI retried only in memory",
        source_decode_warning_counts=dict(Counter(r["decode_warning"] for r in samples)),
        cross_split_exact_duplicates=0,
        cross_split_pixel_duplicates=0,
        cross_split_group_overlap=0,
    )
    (out / "manifests/summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
