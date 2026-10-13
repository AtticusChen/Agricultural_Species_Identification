#!/usr/bin/env python3
"""Real V3 website release smoke test. Never intercepts or mocks model requests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ("ground", "aerial", "other")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def concise(data: dict) -> dict:
    return {key: value for key, value in data.items() if key != "annotated_image"}


def stable_screenshot(page, path: Path, *, full_page: bool = True) -> None:
    page.evaluate("window.scrollTo({top: 0, left: 0, behavior: 'instant'})")
    page.screenshot(path=str(path), full_page=full_page, animations="disabled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--phase", choices=("baseline", "preview", "production"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = args.url.rstrip("/")
    report: dict = {
        "scope": (
            "REAL SERVER V3 MODEL, no API mocks; functional regression, not accuracy benchmark"
        ),
        "phase": args.phase,
        "url": base,
        "samples": {},
    }
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=CHROME)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000}, reduced_motion="reduce", accept_downloads=True
        )
        health = context.request.get(base + "/api/v1/health").json()
        assert health["status"] == "ready", health
        assert health["model"] == "mikania-ensemble-v3", health
        assert health["default_threshold"] == 0.88, health
        report["health"] = health

        if args.phase == "baseline":
            for sample in SAMPLES:
                path = ROOT / "src/mikania_app/static/media" / f"example-{sample}.jpg"
                data = path.read_bytes()
                response = context.request.post(
                    base + "/api/v1/predict",
                    multipart={
                        "file": {"name": path.name, "mimeType": "image/jpeg", "buffer": data}
                    },
                    timeout=180000,
                )
                assert response.ok, response.text()[:500]
                result = response.json()
                report["samples"][sample] = {
                    "source_sha256": hashlib.sha256(data).hexdigest(),
                    "result": concise(result),
                }
        else:
            assert args.baseline is not None, (
                "Preview/production must compare against a saved pre-release baseline"
            )
            baseline = json.loads(args.baseline.read_text())
            errors: list[str] = []
            network_errors: list[str] = []
            responses: list[dict] = []
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "response",
                lambda response: (
                    network_errors.append(f"{response.status} {response.url}")
                    if response.status >= 400
                    else None
                ),
            )
            page.goto(base)
            page.wait_for_load_state("networkidle")
            expect(page.locator("#healthIndicator")).to_have_attribute("data-state", "ready")
            assert page.locator("#workspaceView").is_visible()
            assert page.locator("#startButton").is_disabled()
            expect(page.locator("#thresholdValue")).to_have_text("88.0%")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            for img in page.locator(".sample-grid img").all():
                assert img.evaluate("img => img.complete && img.naturalWidth > 0")
            stable_screenshot(page, args.output / "desktop-home.png")
            for sample in SAMPLES:
                page.locator(f'[data-sample="{sample}"]').click()
                expect(page.locator("#startButton")).to_be_enabled(timeout=30000)
            expect(page.locator(".queue-item")).to_have_count(3)
            page.on(
                "response",
                lambda response: (
                    responses.append(response.json())
                    if response.url.endswith("/api/v1/predict") and response.ok
                    else None
                ),
            )
            page.locator("#startButton").click()
            expect(page.locator('.queue-item[data-status="done"]')).to_have_count(3, timeout=180000)
            assert len(responses) == 3
            for sample, result in zip(SAMPLES, responses, strict=True):
                expected = baseline["samples"][sample]["result"]
                for field in (
                    "present",
                    "count",
                    "threshold",
                    "scan_mode",
                    "model",
                    "image_width",
                    "image_height",
                    "detections",
                ):
                    assert result[field] == expected[field], f"Regression for {sample}: {field}"
                report["samples"][sample] = {"result": concise(result), "matches_pre_release": True}
            page.locator(".queue-select").first.click()
            expect(page.locator("#resultContent")).to_be_visible()
            assert page.locator("#statScore").inner_text() == "—", (
                "Legacy API must not invent a photo-level score"
            )
            assert page.locator("#statRegions").inner_text() == str(responses[0]["count"])
            page.locator('[data-display="compare"]').click()
            expect(page.locator("#originalFrame")).to_be_visible()
            expect(page.locator("#annotatedFrame")).to_be_visible()
            stable_screenshot(page, args.output / "desktop-real-comparison.png")
            page.locator("#zoomButton").click()
            expect(page.locator("#imageDialog")).to_be_visible()
            assert page.locator("#dialogImage").evaluate(
                "img => img.complete && img.naturalWidth > 0"
            )
            page.keyboard.press("Escape")
            expect(page.locator("#imageDialog")).not_to_be_visible()
            with page.expect_download() as download_info:
                page.locator("#downloadImage").click()
            image_path = args.output / "real-result-download.jpg"
            download_info.value.save_as(image_path)
            assert image_path.read_bytes() == base64.b64decode(
                responses[0]["annotated_image"].split(",", 1)[1]
            )
            with page.expect_download() as download_info:
                page.locator("#exportCsv").click()
            csv_path = args.output / "real-session.csv"
            download_info.value.save_as(csv_path)
            assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
            page.locator('.main-nav [data-view="history"]').click()
            expect(page.locator("#historyTableBody tr")).to_have_count(3)
            stable_screenshot(page, args.output / "desktop-real-history.png")
            page.locator('.main-nav [data-view="guide"]').click()
            stable_screenshot(page, args.output / "desktop-guide.png")
            page.locator('.main-nav [data-view="workspace"]').click()
            page.set_viewport_size({"width": 1280, "height": 800})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            stable_screenshot(page, args.output / "laptop-real-comparison.png")
            laptop = context.new_page()
            laptop.set_viewport_size({"width": 1280, "height": 800})
            laptop.goto(base)
            laptop.wait_for_load_state("networkidle")
            stable_screenshot(laptop, args.output / "laptop-home.png", full_page=False)
            report["laptop_start_button_bounds"] = laptop.locator("#startButton").bounding_box()
            assert not errors, errors
            assert not network_errors, network_errors
            report["browser_errors"] = errors
            report["http_errors"] = network_errors
            report["real_ui_checks"] = [
                "three real samples and sequential requests",
                "same V3 decisions and every region as old service",
                "missing score rendered as unavailable",
                "original/result comparison",
                "large image and Escape",
                "download bytes match actual response",
                "UTF-8 BOM CSV",
                "three history records",
                "guide navigation",
                "desktop/laptop no horizontal overflow",
            ]
        report["status"] = "passed"
        (args.output / "live-test-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        browser.close()
    print(
        json.dumps(
            {
                "status": "passed",
                "phase": args.phase,
                "report": str(args.output / "live-test-report.json"),
                "samples": {
                    key: {"present": value["result"]["present"], "count": value["result"]["count"]}
                    for key, value in report["samples"].items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
