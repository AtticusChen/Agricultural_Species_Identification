"""Self-contained offline desktop entry. No system Python or network download."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_ID = "mikania-offline-v4.1.0"
HOST = "127.0.0.1"


def log_directory() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local/share")))
    target = base / "Mikania" / "4.1" / "logs"
    target.mkdir(parents=True, exist_ok=True)
    return target


def configure() -> dict:
    config = json.loads((ROOT / "models/model.json").read_text(encoding="utf-8"))
    export = json.loads((ROOT / "models/export.json").read_text(encoding="utf-8"))
    for branch in ("scene", "patch"):
        path = ROOT / "models" / f"{branch}.onnx"
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != export["models"][branch]["sha256"]
        ):
            raise RuntimeError(f"{branch} 模型缺失或校验失败。请完整解压整个ZIP，勿只复制EXE。")
    if config != export["config"]:
        raise RuntimeError("模型参数与导出记录不一致，请重新解压完整包。")
    # A user's unrelated development variables must not change this frozen release.
    for name in list(os.environ):
        if name.startswith("MIKANIA_"):
            del os.environ[name]
    os.environ.update(
        {
            "MIKANIA_MODEL_TASK": "ensemble",
            "MIKANIA_MODEL_ID": config["model_id"],
            "MIKANIA_DEPLOYMENT_ID": DEPLOYMENT_ID,
            "MIKANIA_MODEL_PATH": str(ROOT / "models/patch.onnx"),
            "MIKANIA_SCENE_MODEL_PATH": str(ROOT / "models/scene.onnx"),
            "MIKANIA_DEVICE": "cpu",
            "MIKANIA_DEFAULT_SCAN_MODE": "precise",
            "MIKANIA_DECISION_METHOD": config["decision_method"],
            "MIKANIA_PRECISE_THRESHOLD": str(config["threshold"]),
            "MIKANIA_FAST_THRESHOLD": str(config["threshold"]),
            "MIKANIA_CLASSIFIER_IMAGE_SIZE": str(config["patch_image_size"]),
            "MIKANIA_SCENE_IMAGE_SIZE": str(config["scene_image_size"]),
            "MIKANIA_CLASSIFIER_TILE_SIZE": str(config["tile_size"]),
            "MIKANIA_CLASSIFIER_TILE_OVERLAP": str(config["overlap"]),
            "MIKANIA_SCENE_PAD_SQUARE": "1",
            "MIKANIA_PATCH_PAD_SQUARE": "1",
            "MIKANIA_PAD_SIZE": str(config["pad_size"]),
            "MIKANIA_REGION_THRESHOLD": str(config["region_threshold"]),
        }
    )
    sys.path.insert(0, str(ROOT / "app"))
    return config


def existing_app(port: int, config: dict) -> bool:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{HOST}:{port}/api/v1/health", timeout=1) as response:
            payload = json.load(response)
        return (
            payload.get("status") == "ready"
            and payload.get("model") == config["model_id"]
            and payload.get("default_threshold") == config["threshold"]
            and payload.get("deployment_id") == DEPLOYMENT_ID
        )
    except (OSError, ValueError):
        return False


def ready_browser(port: int, config: dict) -> None:
    for _ in range(120):
        if existing_app(port, config):
            webbrowser.open(f"http://{HOST}:{port}")
            return
        time.sleep(0.5)
    logging.error("自动打开浏览器等待超时，请查看窗口中的本机地址。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    folder = log_directory()
    logging.basicConfig(
        filename=folder / "launcher.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    print("薇甘菊识别系统 V4.1 · 本机离线版", flush=True)
    print(f"日志：{folder}", flush=True)
    try:
        config = configure()
        import uvicorn
        from PIL import Image

        from mikania_app.main import app

        print("正在校验并加载两个模型，请稍候……", flush=True)
        detector = app.state.detector
        prediction = detector.predict_result(
            Image.new("RGB", (96, 64), "#68834d"), config["threshold"]
        )
        self_test = {
            "status": "passed",
            "model": detector.model_name,
            "threshold": config["threshold"],
            "python": sys.version,
            "cpu_sample_score": prediction.score,
            "root": str(ROOT),
        }
        (folder / "self-test.json").write_text(
            json.dumps(self_test, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if args.self_test:
            print(json.dumps(self_test, ensure_ascii=False), flush=True)
            return 0
        for port in range(18041, 18051):
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if os.name == "nt":
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                connection.bind((HOST, port))
            except OSError:
                connection.close()
                if existing_app(port, config):
                    print(f"识别系统已在运行：http://{HOST}:{port}", flush=True)
                    if not args.no_browser:
                        webbrowser.open(f"http://{HOST}:{port}")
                    return 0
                continue
            connection.listen(128)
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=HOST,
                    port=port,
                    log_level="warning",
                    access_log=False,
                    loop="asyncio",
                    http="h11",
                )
            )
            if not args.no_browser:
                threading.Thread(target=ready_browser, args=(port, config), daemon=True).start()
            print(
                f"本机地址：http://{HOST}:{port}\n请保持本窗口开启；关闭窗口停止系统。", flush=True
            )
            print(config["warning"], flush=True)
            try:
                server.run(sockets=[connection])
            finally:
                connection.close()
            return 0
        raise RuntimeError("18041–18050端口均不可用，请关闭旧窗口后重试。")
    except Exception:
        logging.exception("启动失败")
        (folder / "last-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"启动失败。请将 {folder / 'last-error.txt'} 发给维护人员。", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
