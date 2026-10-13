from __future__ import annotations

from pathlib import Path

from PIL import Image

from mikania_data.inventory import BadImage, ImageRecord
from mikania_data.preview import create_contact_sheets
from mikania_data.selection import (
    PilotConfig,
    diverse_group_sample,
    materialize_pilot,
    select_pilot,
)


def _record(path: str, cohort: str, modality: str, group: str, value: float) -> ImageRecord:
    return ImageRecord(
        relative_path=path,
        cohort=cohort,
        modality=modality,  # type: ignore[arg-type]
        candidate_label="candidate",
        width=64,
        height=64,
        file_bytes=100,
        captured_at="",
        camera="",
        has_gps=False,
        group_id=group,
        sha256=f"{int(value * 100):064x}",
        dhash=f"{int(value * 100):016x}",
        feature=(value, 1.0 - value),
    )


def test_diverse_sample_is_deterministic_and_group_aware() -> None:
    records = [
        _record(f"{index}.jpg", "__root__", "ground", f"group-{index}", index / 10)
        for index in range(6)
    ]
    first = diverse_group_sample(records, 4, seed=17)
    second = diverse_group_sample(records, 4, seed=17)
    assert [item.relative_path for item in first] == [item.relative_path for item in second]
    assert len({item.group_id for item in first}) == 4


def test_select_and_materialize_pilot(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "无人机").mkdir()
    Image.new("RGB", (64, 64), "green").save(data / "positive.jpg")
    Image.new("RGB", (64, 64), "white").save(data / "无人机" / "uav.jpg")
    records = [
        _record("positive.jpg", "__root__", "ground", "ground-1", 0.2),
        _record("无人机/uav.jpg", "无人机", "uav", "uav-1", 0.8),
    ]
    config = PilotConfig(
        seed=1,
        group_window_minutes=5,
        ground_quotas={"__root__": 1},
        uav_count=1,
        link_mode="symlink",
    )

    selected = select_pilot(records, config)
    output = materialize_pilot(
        selected,
        data,
        tmp_path / "pilot",
        "symlink",
        excluded_bad_images=[BadImage("unreadable.jpg", "truncated")],
    )

    assert (output / "manifest.csv").is_file()
    assert (output / "excluded_bad_images.csv").is_file()
    assert len(list((output / "images" / "ground").iterdir())) == 1
    assert len(list((output / "images" / "uav").iterdir())) == 1
    assert all(path.is_symlink() for path in (output / "images").glob("*/*"))

    pages = create_contact_sheets(
        output / "manifest.csv", output, tmp_path / "preview", per_page=1
    )
    assert len(pages) == 2
    assert all(path.is_file() for path in pages)
