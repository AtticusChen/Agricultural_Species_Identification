from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_yolo_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mikania-train", description="运行可追溯的薇甘菊 YOLO 检测实验"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = _parser().parse_args()
    record = run_yolo_experiment(args.config, args.project_root)
    print(f"Experiment record written to {record}")


if __name__ == "__main__":
    main()

