"""Prepare local website samples without touching the source image collection.

This is a reproducible asset build, not a source-data or training-data rewrite.
The full-size examples retain their oriented pixel dimensions. All saved JPEGs
are rebuilt from RGB pixels so GPS, EXIF, XMP, comments, and embedded thumbnails
are not copied into the publicly served assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "ground": "data/IMG_20250228_140606.jpg",
    "aerial": "data/airport/DJI_0418.JPG",
    "other": "data/其他/IMG_20251119_135435.jpg",
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_pixels(source: Path, example_id: str) -> tuple[Image.Image, dict[str, object]]:
    with Image.open(source) as image:
        exif = image.getexif()
        metadata: dict[str, object] = {
            "source_dimensions": list(image.size),
            "source_sha256": sha256(source),
            "source_metadata_keys": sorted(image.info),
            "source_had_exif": bool(exif),
            "source_had_gps": 34853 in exif,
            "source_orientation": exif.get(274, 1),
            "privacy_redactions": [],
        }
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        pixels = Image.frombytes("RGB", oriented.size, oriented.tobytes())
    metadata["oriented_dimensions"] = list(pixels.size)
    if example_id == "ground":
        # This known source has a white camera footer below the plant photograph.
        # Its bottom text line contains visibly burned-in GPS coordinates. Cover
        # only that line, keeping the photograph and full pixel dimensions intact.
        expected_dimensions = (4096, 3544)
        if pixels.size != expected_dimensions:
            raise ValueError("Ground example changed; re-inspect its privacy footer first")
        redactions = [
            {"box_xyxy": [0, 3320, 2350, 3544], "reason": "burned-in date footer text"},
            {"box_xyxy": [2680, 3320, 4096, 3544], "reason": "burned-in GPS footer text"},
        ]
        for redaction in redactions:
            ImageDraw.Draw(pixels).rectangle(redaction["box_xyxy"], fill="white")
        metadata["privacy_redactions"] = redactions
    return pixels, metadata


def save_asset(image: Image.Image, path: Path, quality: int) -> dict[str, object]:
    clean = Image.frombytes("RGB", image.size, image.tobytes())
    clean.save(path, format="JPEG", quality=quality, optimize=True, progressive=True)
    with Image.open(path) as saved:
        if saved.getexif() or any(
            key in saved.info
            for key in ("exif", "xmp", "XML:com.adobe.xmp", "comment", "icc_profile")
        ):
            raise ValueError(f"Metadata remained in generated asset: {path.name}")
    return {
        "file": path.name,
        "dimensions": list(image.size),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "exif_gps_xmp_removed": True,
    }


def build_media(project_root: Path, output: Path, manifest: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for example_id, source_relative in SOURCES.items():
        source = project_root / source_relative
        source_hash = sha256(source)
        pixels, metadata = prepare_pixels(source, example_id)
        metadata.update({"id": example_id, "source": source_relative})
        metadata["example"] = save_asset(pixels, output / f"example-{example_id}.jpg", 95)
        thumbnail = ImageOps.fit(pixels, (360, 240), method=Image.Resampling.LANCZOS)
        metadata["thumbnail"] = save_asset(thumbnail, output / f"thumb-{example_id}.jpg", 84)
        if example_id == "aerial":
            hero = ImageOps.fit(
                pixels,
                (1600, 700),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.65),
            )
            metadata["hero"] = save_asset(hero, output / "hero.jpg", 88)
            metadata["hero_use"] = "Decorative real-scene photograph; no detection marks"
        if sha256(source) != source_hash:
            raise ValueError(f"Source changed during asset preparation: {source_relative}")
        records.append(metadata)
    result: dict[str, object] = {
        "schema_version": 1,
        "purpose": "Public website examples and decorative banner; not training data",
        "metadata_policy": "Rebuild from oriented RGB pixels; omit all source metadata",
        "originals_modified": False,
        "records": records,
    }
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "src/mikania_app/static/media"
    )
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "artifacts/web-redesign/media-manifest.json"
    )
    args = parser.parse_args()
    result = build_media(args.project_root, args.output, args.manifest)
    print(json.dumps({"assets": len(result["records"]), "manifest": str(args.manifest)}))


if __name__ == "__main__":
    main()
