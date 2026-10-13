from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/open_mikania_app.command"
ENTRY = ROOT / "打开薇甘菊识别.command"


def test_launcher_shell_syntax() -> None:
    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("macOS launcher requires zsh")
    for script in (ENTRY, LAUNCHER):
        subprocess.run([zsh, "-n", str(script)], check=True, capture_output=True, timeout=10)


def test_launcher_uses_server_password_without_local_key_prompt() -> None:
    ssh = shutil.which("ssh")
    if ssh is None:
        pytest.skip("OpenSSH is not installed")
    source = LAUNCHER.read_text()
    ssh_command = source.split("  /usr/bin/ssh \\\n", 1)[1].split('    "${REMOTE_HOST}"', 1)[0]
    options = re.findall(r"-o\s+([A-Za-z]+=[^\s\\]+)", ssh_command)
    args = [ssh, "-G", "-F", "/dev/null"]
    for option in options:
        args.extend(["-o", option])
    # -G only resolves configuration; it never connects or starts a tunnel.
    result = subprocess.run(
        [*args, "cx@10.33.231.250"], check=True, capture_output=True, text=True, timeout=10
    )
    settings = dict(line.split(" ", 1) for line in result.stdout.splitlines() if " " in line)
    assert settings["preferredauthentications"] == "password"
    assert settings["pubkeyauthentication"] in {"false", "no"}
    assert settings["exitonforwardfailure"] in {"true", "yes"}
    assert settings["serveraliveinterval"] == "30"
    assert "服务器登录密码" in source
