from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import DetectionDatasetConfig, build_detection_dataset
from .inventory import scan_dataset, write_audit
from .preview import create_contact_sheets
from .sanitize import sanitize_detection_jpegs
from .selection import PilotConfig, materialize_pilot, select_pilot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mikania-data", description="薇甘菊数据审计与试标样本准备工具"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="扫描数据并生成不可变版本的审计清单")
    audit.add_argument("--data", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    audit.add_argument("--group-window-minutes", type=int, default=5)

    pilot = subparsers.add_parser("pilot", help="生成分组感知、视觉多样的试标包")
    pilot.add_argument("--data", required=True, type=Path)
    pilot.add_argument("--config", required=True, type=Path)
    pilot.add_argument("--output", required=True, type=Path)

    preview = subparsers.add_parser("preview", help="从试标清单生成分页缩略总览")
    preview.add_argument("--manifest", required=True, type=Path)
    preview.add_argument("--pilot-root", required=True, type=Path)
    preview.add_argument("--output", required=True, type=Path)

    build_detection = subparsers.add_parser(
        "build-detection", help="生成分组感知、可追溯的 YOLO 检测数据集"
    )
    build_detection.add_argument("--raw-data", required=True, type=Path)
    build_detection.add_argument("--negative-data", required=True, type=Path)
    build_detection.add_argument("--config", required=True, type=Path)
    build_detection.add_argument("--output", required=True, type=Path)

    sanitize_jpeg = subparsers.add_parser(
        "sanitize-jpeg", help="复制检测数据集并无损补齐 JPEG 结束标记"
    )
    sanitize_jpeg.add_argument("--dataset", required=True, type=Path)
    sanitize_jpeg.add_argument("--output", required=True, type=Path)
    sanitize_jpeg.add_argument("--version", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "audit":
        records, bad_images = scan_dataset(args.data, args.group_window_minutes)
        output = write_audit(records, bad_images, args.output)
        print(f"Audit written to {output} ({len(records)} images, {len(bad_images)} bad)")
        return

    if args.command == "pilot":
        config = PilotConfig.from_json(args.config)
        records, bad_images = scan_dataset(args.data, config.group_window_minutes)
        if bad_images:
            print(
                f"Warning: excluding {len(bad_images)} unreadable image(s); "
                "see excluded_bad_images.csv"
            )
        selected = select_pilot(records, config)
        output = materialize_pilot(
            selected,
            args.data,
            args.output,
            config.link_mode,
            excluded_bad_images=bad_images,
        )
        print(f"Pilot written to {output} ({len(selected)} images)")
        return

    if args.command == "preview":
        pages = create_contact_sheets(args.manifest, args.pilot_root, args.output)
        print(f"Preview written to {args.output} ({len(pages)} pages)")
        return

    if args.command == "build-detection":
        config = DetectionDatasetConfig.from_json(args.config)
        output = build_detection_dataset(
            args.raw_data,
            args.negative_data,
            args.output,
            config,
        )
        print(f"Detection dataset written to {output}")
        return

    if args.command == "sanitize-jpeg":
        output = sanitize_detection_jpegs(args.dataset, args.output, args.version)
        print(f"Sanitized detection dataset written to {output}")
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
