from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIRTUALENV = ROOT / "vendor" / "virtualenv.pyz"
ENVIRONMENT = ROOT / ".venv"
VIRTUALENV_SHA256 = "d63ac10e610738c3dcfa51ee0799e6b69ce2964d76cab734542684baf0ed3a5f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment_python() -> Path:
    if sys.platform == "win32":
        return ENVIRONMENT / "Scripts" / "python.exe"
    return ENVIRONMENT / "bin" / "python"


def _environment_is_ready() -> bool:
    python = _environment_python()
    if not python.is_file():
        return False
    result = subprocess.run(
        [str(python), "-m", "pip", "--version"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def main() -> int:
    version = sys.version_info[:2]
    if not (3, 10) <= version <= (3, 12):
        print("需要 64 位 Python 3.10、3.11 或 3.12。", file=sys.stderr)
        return 1
    if _environment_is_ready():
        return 0
    if not VIRTUALENV.is_file() or _sha256(VIRTUALENV) != VIRTUALENV_SHA256:
        print("安装组件校验失败，请重新解压便携包。", file=sys.stderr)
        return 1
    if ENVIRONMENT.exists():
        shutil.rmtree(ENVIRONMENT)
    result = subprocess.run(
        [sys.executable, str(VIRTUALENV), str(ENVIRONMENT)],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
