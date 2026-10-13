from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mikania_app import main as app_module


class ExampleDetector:
    ready = True
    model_name = "existing-v3-model"

    def predict(self, image: Image.Image, threshold: float) -> list[object]:
        return []


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.create_app(ExampleDetector()))


def test_examples_schema_is_fixed_order_and_does_not_expose_source_paths(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/examples")
    assert response.status_code == 200
    assert set(response.json()) == {"examples"}
    examples = response.json()["examples"]
    assert [example["id"] for example in examples] == ["ground", "aerial", "other"]
    for example in examples:
        assert set(example) == {"id", "title", "url", "thumbnail", "filename"}
        assert example["url"] == f"/assets/media/example-{example['id']}.jpg"
        assert example["thumbnail"] == f"/assets/media/thumb-{example['id']}.jpg"
        assert example["filename"].endswith(".jpg")
    assert "/Users/" not in response.text
    assert "data/" not in response.text


def test_examples_omit_missing_or_symlinked_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    static = tmp_path / "static"
    media = static / "media"
    media.mkdir(parents=True)
    (media / "example-ground.jpg").write_bytes(b"test")
    (media / "thumb-ground.jpg").write_bytes(b"test")
    (media / "example-aerial.jpg").write_bytes(b"test")
    outside = tmp_path / "private.jpg"
    outside.write_bytes(b"private")
    (media / "example-other.jpg").symlink_to(outside)
    (media / "thumb-other.jpg").write_bytes(b"test")
    monkeypatch.setattr(app_module, "STATIC_DIR", static)
    client = TestClient(app_module.create_app(ExampleDetector()))
    assert [item["id"] for item in client.get("/api/v1/examples").json()["examples"]] == ["ground"]
    assert client.get("/assets/media/example-other.jpg").status_code == 404
    assert client.get("/assets/%2e%2e/private.jpg").status_code == 404


def test_all_generated_public_images_are_real_jpegs_without_private_metadata(
    client: TestClient,
) -> None:
    urls = ["/assets/media/hero.jpg"]
    for example in client.get("/api/v1/examples").json()["examples"]:
        urls.extend((example["url"], example["thumbnail"]))
    assert len(urls) == 7
    for url in urls:
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert 0 < len(response.content) < app_module.MAX_UPLOAD_BYTES
        with Image.open(BytesIO(response.content)) as image:
            assert image.format == "JPEG"
            assert not image.getexif()
            assert not {"exif", "xmp", "XML:com.adobe.xmp", "comment", "icc_profile"}.intersection(
                image.info
            )
            if "/thumb-" in url:
                assert image.size == (360, 240)


def test_index_requests_revalidate_but_health_keeps_existing_contract(
    client: TestClient,
) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    payload = client.get("/api/v1/health").json()
    assert set(payload) == {
        "status",
        "model",
        "default_threshold",
        "default_thresholds",
        "default_scan_mode",
    }
    assert payload["status"] == "ready"
    assert payload["model"] == "existing-v3-model"


def test_generated_full_examples_preserve_dimensions_and_source_integrity() -> None:
    import hashlib
    import json

    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "artifacts/web-redesign/media-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("Source audit manifest is only included in the development workspace")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["originals_modified"] is False
    for item in manifest["records"]:
        assert item["example"]["dimensions"] == item["oriented_dimensions"]
        source = root / item["source"]
        if source.is_file():
            with source.open("rb") as handle:
                assert hashlib.file_digest(handle, "sha256").hexdigest() == item["source_sha256"]


def test_pixel_rebuild_drops_gps_and_honors_orientation(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "prepare_web_media", root / "scripts/prepare_web_media.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source.jpg"
    exif = Image.Exif()
    exif[274] = 6
    exif[34853] = {1: "N", 2: (19.0, 20.0, 0.0)}
    Image.new("RGB", (60, 40), "green").save(source, exif=exif)
    pixels, metadata = module.prepare_pixels(source, "other")
    assert pixels.size == (40, 60)
    assert metadata["source_had_gps"] is True
    assert metadata["source_orientation"] == 6
    assert pixels.getexif() == {}
    output = tmp_path / "prepared.jpg"
    module.save_asset(pixels, output, 95)
    with Image.open(output) as image:
        assert image.size == (40, 60)
        assert not image.getexif()
