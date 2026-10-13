#!/usr/bin/env python3
"""Render coarse multi-region evidence maps from a patch classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_classifiers_v3 import mikania_probability, tile_starts
from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, required=True)
    return parser.parse_args()


def percentile_75(scores: list[float]) -> float:
    ordered = sorted(scores)
    position = (len(ordered) - 1) * 0.75
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    report: list[dict[str, object]] = []
    for image_path in args.image:
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        tiles = [
            (
                left,
                top,
                min(image.width, left + args.tile_size),
                min(image.height, top + args.tile_size),
            )
            for top in tile_starts(image.height, args.tile_size, args.overlap)
            for left in tile_starts(image.width, args.tile_size, args.overlap)
        ]
        results = model.predict(
            source=[image.crop(box) for box in tiles],
            imgsz=args.image_size,
            device=args.device,
            batch=64,
            verbose=False,
        )
        scores = [mikania_probability(result) for result in results]
        image_score = percentile_75(scores)
        present = image_score >= args.threshold

        canvas = image.convert("RGBA")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        line_width = max(4, round(min(image.size) / 250))
        regions: list[dict[str, object]] = []
        for box, score in zip(tiles, scores, strict=True):
            if score < args.threshold:
                continue
            alpha = min(110, 35 + round(75 * score))
            draw.rectangle(
                box,
                fill=(36, 196, 112, alpha),
                outline=(0, 110, 60, 230),
                width=line_width,
            )
            regions.append({"box_xyxy": box, "score": score})
        canvas = Image.alpha_composite(canvas, overlay)
        banner_height = max(72, round(image.height * 0.045))
        banner = ImageDraw.Draw(canvas)
        banner.rectangle((0, 0, image.width, banner_height), fill=(15, 26, 23, 225))
        label = (
            f"Mikania evidence: {image_score:.3f} | "
            f"{'PRESENT' if present else 'NOT DETECTED'} | regions: {len(regions)}"
        )
        banner.text(
            (line_width * 3, line_width * 3),
            label,
            fill="white",
            font=ImageFont.load_default(),
        )
        destination = output / f"{image_path.stem}__heatmap.jpg"
        canvas.convert("RGB").save(destination, quality=92, optimize=True)
        report.append(
            {
                "image": str(image_path.resolve()),
                "image_score_q75": image_score,
                "threshold": args.threshold,
                "present": present,
                "region_count": len(regions),
                "regions": regions,
                "output": str(destination),
            }
        )

    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
