from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = RUNTIME_ROOT.parent
MODEL_ID = "mikania-yolo26s-v2"
MODEL_PATH = BUNDLE_ROOT / "模型文件" / MODEL_ID / "model.onnx"
METADATA_PATH = BUNDLE_ROOT / "模型文件" / MODEL_ID / "metadata.json"
APP_PATH = BUNDLE_ROOT / "程序源码" / "src"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_model() -> dict[str, object]:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("模型文件不完整，请重新解压便携包。")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    expected = str(metadata["model_sha256"])
    actual = _sha256(MODEL_PATH)
    if actual != expected:
        raise RuntimeError("模型校验失败，请重新复制或解压便携包。")
    return metadata


def _configure_model(metadata: dict[str, object]) -> None:
    thresholds = metadata.get("default_thresholds", {})
    if isinstance(thresholds, dict):
        precise = thresholds.get("precise")
        fast = thresholds.get("fast")
        if isinstance(precise, int | float):
            os.environ["MIKANIA_PRECISE_THRESHOLD"] = str(precise)
        if isinstance(fast, int | float):
            os.environ["MIKANIA_FAST_THRESHOLD"] = str(fast)
    default_mode = metadata.get("default_scan_mode")
    if default_mode in {"precise", "fast"}:
        os.environ["MIKANIA_DEFAULT_SCAN_MODE"] = str(default_mode)


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        try:
            connection.bind((host, port))
        except OSError:
            return False
    return True


def _is_existing_app(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/v1/health", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return payload.get("status") == "ready"


def _open_when_ready(url: str) -> None:
    for _ in range(60):
        if _is_existing_app(url):
            webbrowser.open(url)
            return
        time.sleep(0.5)


def _local_network_urls(port: int) -> list[str]:
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass
    return [f"http://{address}:{port}" for address in sorted(addresses)]


def _self_test() -> int:
    metadata = _verify_model()
    _configure_model(metadata)
    os.environ["MIKANIA_MODEL_PATH"] = str(MODEL_PATH)
    os.environ["MIKANIA_DEVICE"] = "cpu"
    sys.path.insert(0, str(APP_PATH))

    from PIL import Image

    from mikania_app.predictor import OnnxDetector

    detector = OnnxDetector(MODEL_PATH)
    if not detector.ready:
        raise RuntimeError("模型未能正确加载。")
    detector.predict(Image.new("RGB", (64, 64), "#678452"), threshold=0.95)
    print(f"环境和模型检查通过：{detector.model_name}")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    metadata = _verify_model()
    _configure_model(metadata)
    host = os.getenv("MIKANIA_HOST", "127.0.0.1")
    port = int(os.getenv("MIKANIA_PORT", "18080"))
    if not 1 <= port <= 65535:
        raise ValueError("MIKANIA_PORT 必须是 1–65535 之间的端口")
    browser_url = f"http://127.0.0.1:{port}"

    if not _port_is_available(host, port):
        if _is_existing_app(browser_url):
            print(f"识别系统已经运行：{browser_url}")
            webbrowser.open(browser_url)
            return 0
        raise RuntimeError(f"端口 {port} 已被其他程序占用，请更换 MIKANIA_PORT。")

    os.environ["MIKANIA_MODEL_PATH"] = str(MODEL_PATH)
    os.environ["MIKANIA_DEVICE"] = "cpu"
    sys.path.insert(0, str(APP_PATH))

    import uvicorn

    print("\n薇甘菊识别系统已启动。")
    print(f"本机访问地址：{browser_url}")
    if host == "0.0.0.0":
        urls = _local_network_urls(port)
        if urls:
            print("同一局域网内的手机或电脑可打开：")
            for url in urls:
                print(f"  {url}")
        else:
            print("未自动取得局域网地址，请在网络设置中查看本机 IP。")
    print("请保持本窗口开启；关闭窗口即可停止系统。\n")
    threading.Thread(target=_open_when_ready, args=(browser_url,), daemon=True).start()
    uvicorn.run(
        "mikania_app.main:app",
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"启动失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error
