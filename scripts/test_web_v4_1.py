"""Exercise a real V4.1 web service; never mocks inference responses."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright
from test_web_live import CHROME, SAMPLES, concise, stable_screenshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=CHROME)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000}, reduced_motion="reduce", accept_downloads=True
        )
        page = context.new_page()
        errors, responses = [], []
        page.on("pageerror", lambda value: errors.append(str(value)))
        page.on(
            "response",
            lambda response: (
                responses.append(response.json())
                if response.url.endswith("/api/v1/predict") and response.ok
                else None
            ),
        )
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        # Inspect real rendered state before exercising its known controls.
        (args.output / "rendered.html").write_text(page.content())
        stable_screenshot(page, args.output / "home.png")
        health = context.request.get(args.url.rstrip("/") + "/api/v1/health").json()
        assert health["model"] == "mikania-ensemble-v4.1"
        assert health["default_threshold"] == 0.03896419005468488
        expect(page.locator("#healthIndicator")).to_have_attribute("data-state", "ready")
        expect(page.locator("#modelNotice")).to_be_visible()
        expect(page.locator("#thresholdValue")).to_have_text("3.9%")
        for sample in SAMPLES:
            page.locator(f'[data-sample="{sample}"]').click()
            expect(page.locator("#startButton")).to_be_enabled(timeout=30000)
        expect(page.locator(".queue-item")).to_have_count(3)
        page.locator("#startButton").click()
        expect(page.locator('.queue-item[data-status="done"]')).to_have_count(3, timeout=180000)
        assert len(responses) == 3
        for result in responses:
            assert result["model"] == health["model"]
            assert result["threshold"] == health["default_threshold"]
            assert result["present"] == (result["score"] >= result["threshold"])
        page.locator(".queue-select").first.click()
        expect(page.locator("#resultContent")).to_be_visible()
        assert page.locator("#statScore").inner_text() != "—"
        page.locator('[data-display="compare"]').click()
        expect(page.locator("#originalFrame")).to_be_visible()
        expect(page.locator("#annotatedFrame")).to_be_visible()
        stable_screenshot(page, args.output / "comparison.png")
        page.locator("#zoomButton").click()
        expect(page.locator("#imageDialog")).to_be_visible()
        page.keyboard.press("Escape")
        with page.expect_download() as info:
            page.locator("#downloadImage").click()
        downloaded = args.output / "result.jpg"
        info.value.save_as(downloaded)
        assert downloaded.read_bytes() == base64.b64decode(
            responses[0]["annotated_image"].split(",", 1)[1]
        )
        with page.expect_download() as info:
            page.locator("#exportCsv").click()
        info.value.save_as(args.output / "results.csv")
        page.locator('.main-nav [data-view="history"]').click()
        expect(page.locator("#historyTableBody tr")).to_have_count(3)
        page.set_viewport_size({"width": 1280, "height": 800})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator('.main-nav [data-view="workspace"]').click()
        stable_screenshot(page, args.output / "laptop.png")
        assert not errors, errors
        report = {
            "status": "passed",
            "health": health,
            "browser_errors": errors,
            "scope": "Real inference and browser functional test; not accuracy benchmark",
            "samples": {
                sample: concise(result) for sample, result in zip(SAMPLES, responses, strict=True)
            },
        }
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({"status": "passed", "url": args.url, "samples": len(responses)}))
        browser.close()


if __name__ == "__main__":
    main()
