#!/usr/bin/env python3
"""Deploy only the reviewed web release; keep production predictor, weights and unit intact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

PROJECT = Path("/home/cx/Agricultural_Species_Identification")
ARTIFACT_RELATIVE = "artifacts/web-redesign-20260914"
UNIT = Path("/home/cx/.config/systemd/user/mikania-app.service")
BEFORE_MAIN = "ecf560579ccfbac886c03a25d3543d4e93bfdcecd924f8a4f3e50f4a8e2e731a"
PREDICTOR = "e280e8ad5421e4620514e777fd9e00eb2ebb611bf0f3b5189e90b16edb199256"
RELEASE_MAIN = "1c6c2cc561d19b7d638d99f26093663b1681dc175a30f166f70392cde0763a1b"
FILES = (
    "main.py",
    "static/index.html",
    "static/styles.css",
    "static/app.js",
    "static/workspace-core.mjs",
    "static/favicon.svg",
    "static/media/hero.jpg",
    "static/media/example-ground.jpg",
    "static/media/example-aerial.jpg",
    "static/media/example-other.jpg",
    "static/media/thumb-ground.jpg",
    "static/media/thumb-aerial.jpg",
    "static/media/thumb-other.jpg",
)
# Frozen only after the real preview was accepted. Do not derive these hashes at deploy time.
RELEASE_SHA = {
    "main.py": RELEASE_MAIN,
    "static/index.html": "228256e2aac0673b31f837cfeb9ff3e2be62605ac8916e42edb1d8014eba68c3",
    "static/styles.css": "f4b5ac3fe1fbe85d0b756bae98b5f6bf8a4e6883b0f26211d2a1ef326667bbd6",
    "static/app.js": "ee02d31315b27b7098e20ba3a3edcad7b8cbb8417acafd046817e51748ca202b",
    "static/workspace-core.mjs": "a008c04b70f79dd73e172fe98bd2d2b7b911811b05b6bdcf116f6e6c63d1a993",
    "static/favicon.svg": "dff074adca0e407a8c30a38be4716a69feb2e9a848546d4c7d9bc18bca390b0a",
    "static/media/hero.jpg": "90523399996ff25d83d32990dba6bc779982be2d51b1045ab289c07a2a3b50a9",
    "static/media/example-ground.jpg": (
        "10577d196210b2daf7934f5f898304eb3f7c4f1464a16326911578c2f8dfa6a6"
    ),
    "static/media/example-aerial.jpg": (
        "98e0cda9d4cd461d5dc53fb679e45449989eea6a771c50e448575c4da6d5ae32"
    ),
    "static/media/example-other.jpg": (
        "416f184c29b1773150d41ddbc94bfb020fa46b707c56e4d0f5b2370c7dfce31c"
    ),
    "static/media/thumb-ground.jpg": (
        "f89ddd849bf068456e1f900193aaacea0fd0cc4a3fa95a18614513dd5aa8ee5f"
    ),
    "static/media/thumb-aerial.jpg": (
        "0e2ef39046f80ce87960d4b89943a311c8759086a637aa5f6bf11da1e30de16b"
    ),
    "static/media/thumb-other.jpg": (
        "7c040b0efda7150fc4d3b1ba9703221a56ba61a81c9af2184e3afe03365f5dea"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def service(action: str) -> None:
    require(action in {"start", "stop"}, "Unsupported service action")
    completed = subprocess.run(
        ["/usr/bin/systemctl", "--user", action, "mikania-app"],
        capture_output=True,
        text=True,
        timeout=40,
    )
    # On this host, stop returns 1 without diagnostics even when systemd has
    # completed a clean shutdown. Verify the actual state; never infer success
    # from an exit code alone, nor continue copying over a running service.
    observed = subprocess.run(
        [
            "/usr/bin/systemctl",
            "--user",
            "show",
            "mikania-app",
            "--property=ActiveState",
            "--property=MainPID",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    require(observed.returncode == 0, "Cannot verify service state: " + observed.stderr.strip())
    state = dict(line.split("=", 1) for line in observed.stdout.splitlines() if "=" in line)
    expected = "active" if action == "start" else "inactive"
    require(
        state.get("ActiveState") == expected and (action != "stop" or state.get("MainPID") == "0"),
        f"Service did not reach {expected}: {state}; {completed.stderr.strip()}",
    )
    if completed.returncode != 0:
        print(f"systemctl {action} returned {completed.returncode}; verified state is {expected}")


def health() -> dict[str, object]:
    for _ in range(40):
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8000/api/v1/health", timeout=3
            ) as response:
                value = json.load(response)
            require(
                value.get("status") == "ready" and value.get("model") == "mikania-ensemble-v3",
                "Unexpected production model state",
            )
            require(value.get("default_threshold") == 0.88, "Unexpected default threshold")
            require(value.get("default_scan_mode") == "precise", "Unexpected default scan mode")
            return value
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
            time.sleep(0.5)
    raise RuntimeError("Production health did not return the unchanged V3 model and 0.88 threshold")


def copy_atomic(source: Path, target: Path, expected_sha: str | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=target.name + ".web-release-", dir=target.parent, delete=False
    ) as handle:
        temp = Path(handle.name)
    try:
        shutil.copy2(source, temp)
        if expected_sha is not None:
            require(sha(temp) == expected_sha, f"Copy source changed or was corrupted: {source}")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def safe_file(root: Path, name: str, *, missing_ok: bool = False) -> Path:
    require(name in FILES, f"File is outside the release allowlist: {name}")
    require(root == root.resolve(), f"Symlinked root is not allowed: {root}")
    path = root
    for component in Path(name).parts:
        path = path / component
        require(not path.is_symlink(), f"Symlinked release path is not allowed: {path}")
    require(path.resolve().is_relative_to(root), f"Path escaped its release root: {path}")
    require(path.is_file() or (missing_ok and not path.exists()), f"Not a regular file: {path}")
    return path


def unchanged_paths(project: Path, unit: Path) -> tuple[Path, ...]:
    return (
        unit,
        project / "src/mikania_app/predictor.py",
        project / "model_store/mikania-yolo26s-cls-v3/model.pt",
        project / "model_store/mikania-yolo26s-scene-v3/model.pt",
    )


def verify_unchanged(manifest: dict, protected: tuple[Path, ...]) -> None:
    expected = manifest.get("unchanged")
    require(isinstance(expected, dict), "Missing protected-file hashes")
    require(set(expected) == {str(path) for path in protected}, "Invalid protected-file manifest")
    for path in protected:
        require(sha(path) == expected[str(path)], f"Out-of-scope file changed: {path}")


def verify_restore(
    backup: Path,
    target: Path,
    manifest: dict,
    protected: tuple[Path, ...],
) -> None:
    require(isinstance(manifest, dict), "Invalid backup manifest")
    require(manifest.get("schema_version") == 1, "Unsupported backup manifest version")
    entries = manifest.get("files")
    require(isinstance(entries, dict) and set(entries) == set(FILES), "Invalid file allowlist")
    verify_unchanged(manifest, protected)
    for name in FILES:
        info = entries[name]
        require(isinstance(info, dict) and set(info) == {"before", "release"}, "Invalid file entry")
        require(info["release"] == RELEASE_SHA[name], f"Unreviewed release SHA in manifest: {name}")
        before = info["before"]
        require(
            before is None
            or (
                isinstance(before, str)
                and len(before) == 64
                and all(character in "0123456789abcdef" for character in before)
            ),
            f"Invalid original SHA: {name}",
        )
        if before is not None:
            source = safe_file(backup / "app", name)
            require(sha(source) == before, f"Backup file is corrupted: {name}")
        path = safe_file(target, name, missing_ok=True)
        current = sha(path) if path.exists() else None
        # A interrupted release may contain both versions, or not-yet-created new files.
        require(current in {before, info["release"]}, f"File changed outside deployment: {name}")


def restore(
    backup: Path,
    target: Path,
    manifest: dict,
    protected: tuple[Path, ...],
) -> None:
    # Reject unsafe state and corrupt backups before stopping the running service.
    verify_restore(backup, target, manifest, protected)
    service("stop")
    errors: list[str] = []
    try:
        for name in FILES:
            info = manifest["files"][name]
            path = safe_file(target, name, missing_ok=True)
            try:
                if info["before"] is not None:
                    copy_atomic(backup / "app" / name, path, info["before"])
                elif path.exists():
                    require(sha(path) == info["release"], f"New asset was changed: {name}")
                    path.unlink()
            except Exception as error:
                errors.append(f"{name}: {error}")
    finally:
        # Even if one restoration copy fails, do not leave the service deliberately stopped.
        service("start")
    if errors:
        raise RuntimeError("Rollback needs attention: " + "; ".join(errors))
    verify_restore(backup, target, manifest, protected)
    for name in FILES:
        path = target / name
        current = sha(path) if path.exists() else None
        require(current == manifest["files"][name]["before"], f"Rollback was incomplete: {name}")
    health()


def deploy(project: Path, release: Path, backup: Path, unit: Path) -> dict[str, object]:
    target = project / "src/mikania_app"
    require(not backup.exists(), "Never overwrite an existing release backup")
    require(sha(target / "main.py") == BEFORE_MAIN, "Production backend changed since review")
    require(sha(target / "predictor.py") == PREDICTOR, "Production predictor changed since review")
    require(
        sha(release / "predictor.py") == PREDICTOR, "Release predictor is not the server version"
    )
    health()
    protected = unchanged_paths(project, unit)
    manifest = {
        "schema_version": 1,
        "files": {},
        "unchanged": {str(path): sha(path) for path in protected},
    }
    for name in FILES:
        source = safe_file(release, name)
        destination = safe_file(target, name, missing_ok=True)
        require(sha(source) == RELEASE_SHA[name], f"Source is not the accepted preview: {name}")
        manifest["files"][name] = {
            "before": sha(destination) if destination.exists() else None,
            "release": RELEASE_SHA[name],
        }
    backup.mkdir(parents=True)
    shutil.copytree(target, backup / "app", symlinks=True)
    shutil.copy2(unit, backup / "mikania-app.service")
    (backup / "deployment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_restore(backup, target, manifest, protected)
    try:
        service("stop")
        for name in FILES:
            copy_atomic(release / name, target / name, RELEASE_SHA[name])
        service("start")
        current_health = health()
        with urllib.request.urlopen("http://127.0.0.1:8000/", timeout=5) as response:
            require(response.headers.get("Cache-Control") == "no-cache", "Index cache is stale")
            require(b'id="workspaceView"' in response.read(), "New workspace was not served")
        with urllib.request.urlopen("http://127.0.0.1:8000/api/v1/examples", timeout=5) as response:
            require(len(json.load(response)["examples"]) == 3, "Missing release examples")
        verify_unchanged(manifest, protected)
        for name in FILES:
            require(sha(target / name) == RELEASE_SHA[name], f"Release target mismatch: {name}")
    except Exception:
        restore(backup, target, manifest, protected)
        raise
    result = {
        "status": "deployed",
        "health": current_health,
        "backup": str(backup),
        "unchanged_model_predictor_and_service_verified": True,
    }
    (backup / "deploy-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy", "rollback"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--release", type=Path, help="Required only for deploy; unused for rollback"
    )
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    project, backup = args.project.resolve(), args.backup.resolve()
    require(project == PROJECT, "Unexpected project target")
    require(backup == project / ARTIFACT_RELATIVE / "backup-original", "Unexpected backup target")
    target = project / "src/mikania_app"
    unit = UNIT
    require(sha(target / "predictor.py") == PREDICTOR, "Production predictor changed since review")
    if args.action == "rollback":
        manifest = json.loads((backup / "deployment.json").read_text())
        restore(backup, target, manifest, unchanged_paths(project, unit))
        print(json.dumps({"status": "rolled_back", "backup": str(backup)}))
        return
    require(args.release is not None, "--release is required for deploy")
    release = args.release.resolve()
    require(release.is_relative_to(project / ARTIFACT_RELATIVE), "Unexpected release source")
    result = deploy(project, release, backup, unit)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
