from __future__ import annotations

import importlib.util
import json
import shutil
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "web_deploy_test", ROOT / "scripts/deploy_web_release.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    module = load_module()
    project = tmp_path.resolve() / "project"
    target = project / "src/mikania_app"
    release = project / module.ARTIFACT_RELATIVE / "release"
    backup = project / module.ARTIFACT_RELATIVE / "backup-original"
    unit = project / "config/mikania-app.service"
    originals = {}
    original_names = {"main.py", "static/index.html", "static/styles.css", "static/app.js"}
    for name in module.FILES:
        source = release / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"accepted release: {name}".encode())
        if name in original_names:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(f"original production: {name}".encode())
            originals[name] = destination.read_bytes()
    protected = module.unchanged_paths(project, unit)
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"protected: {path.name}".encode())
    shutil.copy2(target / "predictor.py", release / "predictor.py")
    monkeypatch.setattr(module, "BEFORE_MAIN", module.sha(target / "main.py"))
    monkeypatch.setattr(module, "PREDICTOR", module.sha(target / "predictor.py"))
    monkeypatch.setattr(
        module, "RELEASE_SHA", {name: module.sha(release / name) for name in module.FILES}
    )
    calls = []
    monkeypatch.setattr(module, "service", calls.append)
    monkeypatch.setattr(
        module,
        "health",
        lambda: {
            "status": "ready",
            "model": "mikania-ensemble-v3",
            "default_threshold": 0.88,
            "default_scan_mode": "precise",
        },
    )

    def never_call_process(*args, **kwargs):
        raise AssertionError("Tests must never call a real process")

    def fake_urlopen(url, **kwargs):
        data = (
            b'<section id="workspaceView">'
            if url.endswith(":8000/")
            else (json.dumps({"examples": [{}, {}, {}]}).encode())
        )
        response = BytesIO(data)
        response.headers = {"Cache-Control": "no-cache"}
        return response

    monkeypatch.setattr(module.subprocess, "run", never_call_process)
    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    return SimpleNamespace(
        module=module,
        project=project,
        target=target,
        release=release,
        backup=backup,
        unit=unit,
        protected=protected,
        originals=originals,
        calls=calls,
        protected_hashes={path: module.sha(path) for path in protected},
    )


def snapshot(layout: SimpleNamespace) -> dict:
    module = layout.module
    manifest = {
        "schema_version": 1,
        "files": {
            name: {
                "before": module.sha(layout.target / name) if name in layout.originals else None,
                "release": module.RELEASE_SHA[name],
            }
            for name in module.FILES
        },
        "unchanged": {str(path): digest for path, digest in layout.protected_hashes.items()},
    }
    shutil.copytree(layout.target, layout.backup / "app")
    (layout.backup / "deployment.json").write_text(json.dumps(manifest))
    return manifest


def assert_original_state(layout: SimpleNamespace) -> None:
    for name in layout.module.FILES:
        path = layout.target / name
        if name in layout.originals:
            assert path.read_bytes() == layout.originals[name]
        else:
            assert not path.exists()
    assert all(
        layout.module.sha(path) == digest for path, digest in layout.protected_hashes.items()
    )


def test_frozen_source_hashes_match_archived_v3_release() -> None:
    module = load_module()
    assert set(module.RELEASE_SHA) == set(module.FILES)
    for name, digest in module.RELEASE_SHA.items():
        # V3 hashes describe its immutable release, not the current V4.1 working source.
        source = ROOT / "artifacts/web-redesign-20260914/server-release/mikania_app" / name
        assert module.sha(source) == digest, name


def test_complete_deploy_and_rollback_preserve_protected_files(layout: SimpleNamespace) -> None:
    result = layout.module.deploy(layout.project, layout.release, layout.backup, layout.unit)
    assert result["status"] == "deployed"
    manifest = json.loads((layout.backup / "deployment.json").read_text())
    assert all(
        layout.module.sha(layout.target / name) == value["release"]
        for name, value in manifest["files"].items()
    )
    layout.module.restore(layout.backup, layout.target, manifest, layout.protected)
    assert_original_state(layout)
    assert layout.calls == ["stop", "start", "stop", "start"]


@pytest.mark.parametrize("copied_count", [0, 1, 4, 8, 13])
def test_interrupted_partial_release_can_roll_back(
    layout: SimpleNamespace,
    copied_count: int,
) -> None:
    manifest = snapshot(layout)
    for name in layout.module.FILES[:copied_count]:
        destination = layout.target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(layout.release / name, destination)
    layout.module.restore(layout.backup, layout.target, manifest, layout.protected)
    assert_original_state(layout)
    assert layout.calls == ["stop", "start"]


def test_copy_failure_cleans_unique_temp_and_automatically_rolls_back(
    layout: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = layout.module
    original_copy = module.shutil.copy2
    stale = layout.target / "main.py.web-release-tmp"
    stale.write_bytes(b"unrelated previous temporary file")
    failures = []

    def fail_once(source, destination, *args, **kwargs):
        if Path(source) == layout.release / "static/styles.css" and not failures:
            failures.append(str(destination))
            Path(destination).write_bytes(b"partial copy")
            raise OSError("injected disk write failure")
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "copy2", fail_once)
    with pytest.raises(OSError, match="injected disk"):
        module.deploy(layout.project, layout.release, layout.backup, layout.unit)
    assert_original_state(layout)
    assert stale.read_bytes() == b"unrelated previous temporary file"
    assert not Path(failures[0]).exists()
    assert layout.calls == ["stop", "stop", "start"]


def test_corrupt_backup_is_rejected_before_stop(layout: SimpleNamespace) -> None:
    manifest = snapshot(layout)
    (layout.backup / "app/static/index.html").write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="Backup file is corrupted"):
        layout.module.restore(layout.backup, layout.target, manifest, layout.protected)
    assert layout.calls == []
    assert_original_state(layout)


@pytest.mark.parametrize("extra_name", ["../outside.txt", "/etc/anything", "predictor.py"])
def test_manifest_path_escape_or_protected_target_is_rejected(
    layout: SimpleNamespace,
    extra_name: str,
) -> None:
    manifest = snapshot(layout)
    manifest["files"][extra_name] = {"before": None, "release": "0" * 64}
    with pytest.raises(RuntimeError, match="Invalid file allowlist"):
        layout.module.restore(layout.backup, layout.target, manifest, layout.protected)
    assert layout.calls == []
    assert_original_state(layout)


def test_unreviewed_source_is_rejected_before_backup_and_stop(layout: SimpleNamespace) -> None:
    (layout.release / "static/app.js").write_bytes(b"not the accepted preview")
    with pytest.raises(RuntimeError, match="accepted preview"):
        layout.module.deploy(layout.project, layout.release, layout.backup, layout.unit)
    assert not layout.backup.exists()
    assert layout.calls == []
    assert_original_state(layout)


def test_subsequent_edits_are_not_overwritten_by_rollback(layout: SimpleNamespace) -> None:
    manifest = snapshot(layout)
    edited = layout.target / "static/app.js"
    edited.write_bytes(b"later user edit")
    with pytest.raises(RuntimeError, match="changed outside deployment"):
        layout.module.restore(layout.backup, layout.target, manifest, layout.protected)
    assert edited.read_bytes() == b"later user edit"
    assert layout.calls == []


def test_parent_symlink_escape_is_rejected_before_stop(layout: SimpleNamespace) -> None:
    manifest = snapshot(layout)
    static = layout.target / "static"
    static.rename(layout.target / "original-static")
    static.symlink_to(layout.target / "original-static", target_is_directory=True)
    with pytest.raises(RuntimeError, match="Symlinked release path"):
        layout.module.restore(layout.backup, layout.target, manifest, layout.protected)
    assert layout.calls == []


def test_restore_copy_failure_still_attempts_service_start(
    layout: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = snapshot(layout)
    original_copy = layout.module.copy_atomic

    def fail_one(source, target, expected_sha=None):
        if target.name == "main.py":
            raise OSError("injected restore failure")
        return original_copy(source, target, expected_sha)

    monkeypatch.setattr(layout.module, "copy_atomic", fail_one)
    with pytest.raises(RuntimeError, match="Rollback needs attention"):
        layout.module.restore(layout.backup, layout.target, manifest, layout.protected)
    assert layout.calls == ["stop", "start"]


def test_rollback_cli_needs_no_release_directory(
    layout: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot(layout)
    layout.release.rename(layout.release.with_name("staging-no-longer-available"))
    monkeypatch.setattr(layout.module, "PROJECT", layout.project)
    monkeypatch.setattr(layout.module, "UNIT", layout.unit)
    monkeypatch.setattr(
        "sys.argv",
        [
            "deploy_web_release.py",
            "rollback",
            "--project",
            str(layout.project),
            "--backup",
            str(layout.backup),
        ],
    )
    layout.module.main()
    assert_original_state(layout)
    assert layout.calls == ["stop", "start"]


def test_safety_checks_remain_active_in_optimized_python() -> None:
    source = (ROOT / "scripts/deploy_web_release.py").read_text()
    namespace = {"__name__": "_not_main_"}
    exec(compile(source, "deploy_web_release.py", "exec", optimize=2), namespace)
    with pytest.raises(RuntimeError, match="still enforced"):
        namespace["require"](False, "still enforced")


@pytest.mark.parametrize("exit_code", [0, 1])
def test_service_stop_checks_real_stopped_state(monkeypatch, exit_code) -> None:
    module = load_module()
    replies = iter(
        [
            SimpleNamespace(returncode=exit_code, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="ActiveState=inactive\nMainPID=0\n", stderr=""),
        ]
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: next(replies))
    module.service("stop")


@pytest.mark.parametrize(
    "state", ["ActiveState=active\nMainPID=123\n", "ActiveState=inactive\nMainPID=123\n"]
)
def test_service_stop_never_proceeds_with_a_live_process(monkeypatch, state) -> None:
    module = load_module()
    replies = iter(
        [
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout=state, stderr=""),
        ]
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: next(replies))
    with pytest.raises(RuntimeError, match="did not reach inactive"):
        module.service("stop")


def test_service_rejects_unverifiable_state(monkeypatch) -> None:
    module = load_module()
    replies = iter(
        [
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=1, stdout="", stderr="manager unavailable"),
        ]
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: next(replies))
    with pytest.raises(RuntimeError, match="Cannot verify"):
        module.service("stop")
