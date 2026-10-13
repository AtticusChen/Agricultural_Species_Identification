#!/usr/bin/env python3
"""Isolated browser acceptance tests; API fixtures are NOT model validation.

Run with the project's Python Playwright and installed Chrome:
  .venv/bin/python scripts/test_web_redesign.py --url http://127.0.0.1:8765
All API requests are intercepted in an isolated, nonpersistent browser context.
The served HTML/CSS/JavaScript and static example images remain the actual app.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import time
import traceback
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import Page, Route, expect, sync_playwright

DEFAULT_THRESHOLD = 0.880123456789
MODEL_ID = "isolated-browser-fixture-NOT-A-REAL-MODEL"
PROJECT = Path(__file__).resolve().parents[1]


def fixture_image() -> bytes:
    image = Image.new("RGB", (640, 420), "#dce7d8")
    draw = ImageDraw.Draw(image)
    draw.rectangle((95, 85, 420, 320), outline="#0d5637", width=6)
    draw.text((112, 106), "ISOLATED UI TEST - NOT MODEL OUTPUT", fill="#0d5637")
    output = io.BytesIO()
    image.save(output, "JPEG", quality=90)
    return output.getvalue()


IMAGE_BYTES = fixture_image()
IMAGE_DATA = "data:image/jpeg;base64," + base64.b64encode(IMAGE_BYTES).decode()


def upload(name: str, contents: bytes = IMAGE_BYTES, mime: str = "image/jpeg") -> dict:
    return {"name": name, "mimeType": mime, "buffer": contents}


class FixtureAPI:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.health_offline = False
        self.predict_delay = 0.0
        self.attempts: dict[str, int] = {}

    def route(self, route: Route) -> None:
        path = route.request.url.split("?", 1)[0]
        if path.endswith("/health"):
            if self.health_offline:
                route.abort("connectionrefused")
            else:
                route.fulfill(
                    json={
                        "status": "ready",
                        "model": MODEL_ID,
                        "default_threshold": DEFAULT_THRESHOLD,
                        "default_thresholds": {"precise": DEFAULT_THRESHOLD, "fast": 0.5},
                        "default_scan_mode": "precise",
                    }
                )
            return
        if path.endswith("/examples"):
            route.fulfill(
                json={
                    "examples": [
                        {
                            "id": name,
                            "title": name,
                            "filename": f"example-{name}.jpg",
                            "url": f"/api/v1/test-example/{name}",
                            "thumbnail": f"/assets/media/thumb-{name}.jpg",
                        }
                        for name in ("ground", "aerial", "other")
                    ]
                }
            )
            return
        if "/test-example/" in path:
            route.fulfill(body=IMAGE_BYTES, content_type="image/jpeg")
            return
        if not path.endswith("/predict"):
            raise AssertionError(f"Unexpected fixture API route: {path}")
        headers = route.request.headers
        body = route.request.post_data_buffer or b""
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {headers['content-type']}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
        )
        fields = {}
        filename = ""
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "file":
                filename = part.get_filename()
            else:
                fields[name] = part.get_content()
        self.requests.append({"filename": filename, "fields": fields})
        self.attempts[filename] = self.attempts.get(filename, 0) + 1
        if self.predict_delay:
            time.sleep(self.predict_delay)
        if "corrupt" in filename:
            route.fulfill(
                status=400, json={"detail": "无法读取图片，请选择完整的 JPG、PNG 或 WebP 图片"}
            )
            return
        if "flaky" in filename and self.attempts[filename] == 1:
            route.abort("connectionreset")
            return
        positive = "negative" not in filename
        count = 1 if positive and "zero-regions" not in filename else 0
        score = None if "negative" in filename else 0.923456
        route.fulfill(
            json={
                "present": positive,
                "score": score,
                "message": "Fixture result only",
                "count": count,
                "threshold": float(fields.get("threshold", DEFAULT_THRESHOLD)),
                "scan_mode": fields.get("scan_mode", "precise"),
                "model": MODEL_ID,
                "latency_ms": 127.3,
                "image_width": 640,
                "image_height": 420,
                "detections": [
                    {
                        "box": [95, 85, 420, 320],
                        "confidence": 0.98,
                        "class_id": 0,
                        "class_name": "mikania",
                    }
                ]
                if count
                else [],
                "annotated_image": IMAGE_DATA,
            }
        )


INSTRUMENTATION = """(() => {
  const originalFetch = window.fetch.bind(window);
  window.__uiTest = {active: 0, maxActive: 0, created: [], revoked: []};
  window.fetch = async (...args) => {
    const isPrediction = String(args[0]).includes('/api/v1/predict');
    if (isPrediction) {
      window.__uiTest.active++;
      window.__uiTest.maxActive = Math.max(window.__uiTest.maxActive, window.__uiTest.active);
    }
    try { return await originalFetch(...args); }
    finally { if (isPrediction) window.__uiTest.active--; }
  };
  const create = URL.createObjectURL.bind(URL);
  const revoke = URL.revokeObjectURL.bind(URL);
  URL.createObjectURL = value => {
    const url = create(value); window.__uiTest.created.push(url); return url;
  };
  URL.revokeObjectURL = url => { window.__uiTest.revoked.push(url); return revoke(url); };
})();"""


def assert_no_overflow(page: Page) -> None:
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), page.evaluate(
        "({viewport: innerWidth, document: document.documentElement.scrollWidth})"
    )


def stable_screenshot(page: Page, path: Path, full_page: bool = True) -> None:
    page.evaluate("window.scrollTo({top: 0, left: 0, behavior: 'instant'})")
    page.wait_for_function("scrollY === 0 && scrollX === 0")
    page.screenshot(path=str(path), full_page=full_page, animations="disabled")


class Suite:
    def __init__(self, browser, url: str, output: Path) -> None:
        self.browser = browser
        self.url = url
        self.output = output
        self.results: list[dict] = []
        self.page_errors: list[str] = []
        self.layout_metrics: list[dict] = []

    def new_page(self, api: FixtureAPI | None = None):
        api = api or FixtureAPI()
        context = self.browser.new_context(
            viewport={"width": 1440, "height": 1000}, accept_downloads=True
        )
        context.add_init_script(INSTRUMENTATION)
        context.route("**/api/v1/**", api.route)
        page = context.new_page()
        page.set_default_timeout(6000)
        page.on("pageerror", lambda error: self.page_errors.append(str(error)))
        page.goto(self.url)
        page.wait_for_load_state("networkidle")
        return context, page, api

    def record(self, name: str, test) -> None:
        started = time.monotonic()
        try:
            test()
            self.results.append(
                {
                    "name": name,
                    "status": "passed",
                    "duration_seconds": round(time.monotonic() - started, 2),
                }
            )
            print(f"PASS {name}", flush=True)
        except Exception:
            details = traceback.format_exc()
            self.results.append({"name": name, "status": "failed", "details": details})
            print(f"FAIL {name}\n{details}", flush=True)

    def layout_navigation(self) -> None:
        context, page, _ = self.new_page()
        try:
            expect(page.locator("#healthIndicator")).to_have_attribute("data-state", "ready")
            expect(page.locator("#modelName")).to_have_text(MODEL_ID)
            expect(page.locator("#startButton")).to_be_disabled()
            expect(page.locator("#exportCsv")).to_be_disabled()
            expect(page.locator(".workflow-steps li").nth(0)).to_have_attribute(
                "aria-current", "step"
            )
            for width, height in ((1440, 1000), (1280, 800)):
                page.set_viewport_size({"width": width, "height": height})
                assert_no_overflow(page)
                stable_screenshot(page, self.output / f"fixture-empty-{width}.png")
                stable_screenshot(
                    page, self.output / f"fixture-first-screen-{width}.png", full_page=False
                )
                start = page.locator("#startButton").bounding_box()
                self.layout_metrics.append(
                    {
                        "viewport_width": width,
                        "viewport_height": height,
                        "start_button_box": start,
                        "start_button_fully_in_first_screen": (
                            start["y"] >= 0 and start["y"] + start["height"] <= height
                        ),
                    }
                )
                page.locator('[data-view="guide"]').first.click()
                expect(page.locator("#guideView")).to_be_visible()
                assert_no_overflow(page)
                page.locator('[data-view="history"]').first.click()
                expect(page.locator("#historyEmpty")).to_be_visible()
                assert_no_overflow(page)
                page.locator('[data-view="workspace"]').first.click()
            page.locator("#dropZone").focus()
            with page.expect_file_chooser() as chooser:
                page.keyboard.press("Enter")
            chooser.value.set_files(upload("keyboard.jpg"))
            expect(page.locator(".queue-item")).to_have_count(1)
            page.locator(".queue-remove").click()
            expect(page.locator(".queue-item")).to_have_count(0)
            assert page.evaluate(
                "window.__uiTest.created.every(u => window.__uiTest.revoked.includes(u))"
            )
            chooser_events = []
            page.on("filechooser", lambda chooser: chooser_events.append(chooser))
            with page.expect_file_chooser() as chooser:
                page.locator("#chooseFiles").click()
            chooser.value.set_files(upload("button.jpg"))
            expect(page.locator(".queue-item")).to_have_count(1)
            assert len(chooser_events) == 1, "Upload button opened duplicate chooser dialogs."
        finally:
            context.close()

    def predictions_exports_history(self) -> None:
        context, page, api = self.new_page()
        try:
            page.locator("#fileInput").set_input_files(
                [upload("positive.jpg"), upload("zero-regions.jpg"), upload("negative.jpg")]
            )
            expect(page.locator(".queue-item")).to_have_count(3)
            page.locator("#startButton").click()
            expect(page.locator('.queue-item[data-status="done"]')).to_have_count(3)
            expect(page.locator(".workflow-steps li").nth(2)).to_have_attribute(
                "aria-current", "step"
            )
            assert page.evaluate("window.__uiTest.maxActive") == 1
            assert [r["filename"] for r in api.requests] == [
                "positive.jpg",
                "zero-regions.jpg",
                "negative.jpg",
            ]
            assert all("threshold" not in r["fields"] for r in api.requests), api.requests
            assert all(r["fields"].get("scan_mode", "precise") == "precise" for r in api.requests)

            page.locator(".queue-select").filter(has_text="zero-regions.jpg").click()
            expect(page.locator("#resultBadge")).to_have_attribute("data-tone", "positive")
            expect(page.locator("#statRegions")).to_have_text("0")
            expect(page.locator("#resultTitle")).to_contain_text("疑似")
            assert "98" not in page.locator("#statScore").inner_text()
            page.locator('[data-display="compare"]').click()
            expect(page.locator("#originalImage")).to_be_visible()
            expect(page.locator("#annotatedImage")).to_be_visible()
            for width, height in ((1440, 1000), (1280, 800)):
                page.set_viewport_size({"width": width, "height": height})
                assert_no_overflow(page)
                stable_screenshot(page, self.output / f"fixture-result-{width}.png")
            page.locator("#zoomButton").click()
            expect(page.locator("#imageDialog")).to_be_visible()
            expect(page.locator("#dialogImage")).to_have_attribute("src", IMAGE_DATA)
            page.keyboard.press("Escape")
            expect(page.locator("#imageDialog")).not_to_be_visible()
            page.locator("#zoomButton").click()
            page.locator("#closeDialog").click()
            expect(page.locator("#imageDialog")).not_to_be_visible()

            with page.expect_download() as event:
                page.locator("#downloadImage").click()
            download = event.value
            image_path = self.output / "fixture-image-download.jpg"
            download.save_as(image_path)
            assert image_path.read_bytes() == IMAGE_BYTES
            page.locator(".queue-select").filter(has_text="negative.jpg").click()
            expect(page.locator("#resultBadge")).to_have_attribute("data-tone", "negative")
            expect(page.locator("#statScore")).to_have_text("—")
            expect(page.locator("#resultExplanation")).to_contain_text("不等于")

            page.locator('[data-view="history"]').first.click()
            expect(page.locator("#summaryTotal")).to_have_text("3")
            expect(page.locator("#summaryPositive")).to_have_text("2")
            expect(page.locator("#summaryNegative")).to_have_text("1")
            expect(page.locator("#summaryErrors")).to_have_text("0")
            expect(page.locator("#historyTableBody tr")).to_have_count(3)
            page.locator('[data-filter="positive"]').click()
            expect(page.locator("#historyTableBody tr")).to_have_count(2)
            page.locator('[data-filter="negative"]').click()
            expect(page.locator("#historyTableBody tr")).to_have_count(1)
            page.locator('[data-filter="all"]').click()
            page.locator("#historySearch").fill("zero-regions")
            expect(page.locator("#historyTableBody tr")).to_have_count(1)
            page.locator("#historyTableBody .history-open").click()
            expect(page.locator("#workspaceView")).to_be_visible()
            expect(page.locator("#resultFilename")).to_have_text("zero-regions.jpg")
            with page.expect_download() as event:
                page.locator("#exportCsv").click()
            csv_path = self.output / "fixture-results.csv"
            event.value.save_as(csv_path)
            assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
            rows = list(csv.reader(io.StringIO(csv_path.read_text(encoding="utf-8-sig"))))
            assert len(rows) == 4, rows
            assert {row[0] for row in rows[1:]} == {
                "positive.jpg",
                "zero-regions.jpg",
                "negative.jpg",
            }
            page.once("dialog", lambda dialog: dialog.dismiss())
            page.locator("#clearButton").click()
            expect(page.locator(".queue-item")).to_have_count(3)
            page.once("dialog", lambda dialog: dialog.accept())
            page.locator("#clearButton").click()
            expect(page.locator(".queue-item")).to_have_count(0)
            expect(page.locator("[data-display]:disabled")).to_have_count(3)
            expect(page.locator("#zoomButton")).to_be_disabled()
            expect(page.locator("#downloadImage")).to_be_disabled()
            expect(page.locator(".workflow-steps li").nth(0)).to_have_attribute(
                "aria-current", "step"
            )
            page.wait_for_function(
                "window.__uiTest.created.every(u => window.__uiTest.revoked.includes(u))"
            )
        finally:
            context.close()

    def thresholds_reset_formula_csv(self) -> None:
        context, page, api = self.new_page()
        try:
            page.locator("#threshold").evaluate("el => { el.closest('details').open = true; }")
            page.locator("#threshold").evaluate(
                "el => { el.value = '73.7'; "
                "el.dispatchEvent(new Event('input', {bubbles: true})); }"
            )
            page.locator("#fileInput").set_input_files(upload("=SUM(1,1).jpg"))
            page.locator("#startButton").click()
            expect(page.locator('.queue-item[data-status="done"]')).to_have_count(1)
            assert abs(float(api.requests[0]["fields"]["threshold"]) - 0.737) < 1e-12
            with page.expect_download() as event:
                page.locator("#exportCsv").click()
            path = self.output / "fixture-formula-safe.csv"
            event.value.save_as(path)
            rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
            assert rows[1][0].startswith("'=SUM(1,1)"), rows
            page.locator("#resetThreshold").click()
            page.locator("#fileInput").set_input_files(upload("reset-default.jpg"))
            page.locator("#startButton").click()
            expect(page.locator('.queue-item[data-status="done"]')).to_have_count(2)
            assert "threshold" not in api.requests[1]["fields"]
            page.reload()
            page.wait_for_load_state("networkidle")
            expect(page.locator(".queue-item")).to_have_count(0)
            expect(page.locator("#historyCount")).to_have_text("0")
            assert page.evaluate("localStorage.length") == 0
        finally:
            context.close()

    def errors_retry_offline(self) -> None:
        api = FixtureAPI()
        api.health_offline = True
        context, page, api = self.new_page(api)
        try:
            expect(page.locator("#connectionBanner")).to_be_visible()
            expect(page.locator("#healthIndicator")).to_have_attribute("data-state", "offline")
            expect(page.locator("#startButton")).to_be_disabled()
            api.health_offline = False
            page.locator("#reconnectButton").click()
            expect(page.locator("#healthIndicator")).to_have_attribute("data-state", "ready")
            expect(page.locator("#connectionBanner")).not_to_be_visible()
            page.locator("#fileInput").set_input_files(
                [upload("corrupt.jpg", b"NOT AN IMAGE"), upload("flaky.jpg")]
            )
            page.locator("#startButton").click()
            expect(page.locator('.queue-item[data-status="error"]')).to_have_count(2)
            page.locator(".queue-select").filter(has_text="corrupt.jpg").click()
            expect(page.locator("#resultBadge")).to_have_attribute("data-tone", "error")
            assert "未检出薇甘菊" not in page.locator("#resultTitle").inner_text()
            expect(page.locator("#connectionBanner")).to_be_visible()
            page.locator("#reconnectButton").click()
            expect(page.locator("#healthIndicator")).to_have_attribute("data-state", "ready")
            page.locator("#startButton").click()
            expect(page.locator('.queue-item[data-status="done"]')).to_have_count(1)
            expect(page.locator('.queue-item[data-status="error"]')).to_have_count(1)
            assert api.attempts == {"corrupt.jpg": 2, "flaky.jpg": 2}, api.attempts
            page.locator('[data-view="history"]').first.click()
            expect(page.locator("#summaryErrors")).to_have_text("1")
            page.locator('[data-filter="error"]').click()
            expect(page.locator("#historyTableBody tr")).to_have_count(1)
            expect(page.locator("#historyTableBody")).to_contain_text("corrupt.jpg")
            stable_screenshot(page, self.output / "fixture-error-history.png")
        finally:
            context.close()

    def limits_examples_pause(self) -> None:
        api = FixtureAPI()
        api.predict_delay = 0.25
        context, page, api = self.new_page(api)
        try:
            page.locator("#fileInput").set_input_files(
                [upload("empty.jpg", b""), upload("notes.txt", b"text", "text/plain")]
            )
            expect(page.locator(".queue-item")).to_have_count(0)
            expect(page.locator("#notice")).not_to_have_text("")
            page.locator('[data-sample="ground"]').click()
            expect(page.locator(".queue-item")).to_have_count(1)
            expect(page.locator(".queue-name")).to_have_text("example-ground.jpg")
            assert not api.requests, "Examples must not silently run inference."
            page.locator("#fileInput").set_input_files(
                [upload(f"queued-{i}.jpg") for i in range(11)]
            )
            expect(page.locator(".queue-item")).to_have_count(10)
            page.evaluate("""() => {
                const timer = setInterval(() => {
                    const pause = document.querySelector('#pauseButton');
                    if (!document.querySelector('.queue-item[data-status="running"]')
                        || pause.hidden) return;
                    window.__busyControlsDisabled = ['chooseFiles', 'addFiles',
                        'clearButton', 'fileInput', 'threshold'].every(
                        id => document.getElementById(id).disabled);
                    pause.click();
                    clearInterval(timer);
                }, 10);
            }""")
            page.locator("#startButton").click()
            expect(page.locator('.queue-item[data-status="done"]')).to_have_count(1)
            expect(page.locator('.queue-item[data-status="pending"]')).to_have_count(9)
            assert len(api.requests) == 1
            assert page.evaluate("window.__busyControlsDisabled") is True
            page.locator("#startButton").click()
            expect(page.locator('.queue-item[data-status="done"]')).to_have_count(10)
            assert len(api.requests) == 10
            assert page.evaluate("window.__uiTest.maxActive") == 1
        finally:
            context.close()

    def upload_limits_dedup_and_safe_names(self) -> None:
        context, page, _ = self.new_page()
        try:
            page.evaluate(
                """encoded => {
                    const input = document.querySelector('#fileInput');
                    const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
                    const file = new File([bytes], 'same.jpg',
                        {type: 'image/jpeg', lastModified: 1234567890000});
                    for (let i = 0; i < 2; i++) {
                        const transfer = new DataTransfer();
                        transfer.items.add(file);
                        input.files = transfer.files;
                        input.dispatchEvent(new Event('change', {bubbles: true}));
                    }
                }""",
                base64.b64encode(IMAGE_BYTES).decode(),
            )
            expect(page.locator(".queue-item")).to_have_count(1)
            page.locator("#fileInput").set_input_files(upload("<img onerror=alert(1)>.jpg"))
            expect(page.locator(".queue-item")).to_have_count(2)
            expect(page.locator(".queue-name").last).to_have_text("<img onerror=alert(1)>.jpg")
            assert page.locator("#queueList img[onerror]").count() == 0
            page.locator("#fileInput").set_input_files(upload("empty-mime.jpg", mime=""))
            expect(page.locator(".queue-item")).to_have_count(3)
            page.evaluate("""() => {
                const input = document.querySelector('#fileInput');
                const transfer = new DataTransfer();
                transfer.items.add(new File([new Uint8Array(21 * 1024 * 1024)],
                    'too-large.jpg', {type: 'image/jpeg'}));
                input.files = transfer.files;
                input.dispatchEvent(new Event('change', {bubbles: true}));
            }""")
            expect(page.locator(".queue-item")).to_have_count(3)
            page.locator("#clearButton").click()
            expect(page.locator(".queue-item")).to_have_count(0)
            page.evaluate("""() => {
                const input = document.querySelector('#fileInput');
                const transfer = new DataTransfer();
                for (let i = 0; i < 5; i++) {
                    transfer.items.add(new File([new Uint8Array(17 * 1024 * 1024)],
                        `budget-${i}.jpg`, {type: 'image/jpeg'}));
                }
                input.files = transfer.files;
                input.dispatchEvent(new Event('change', {bubbles: true}));
            }""")
            expect(page.locator(".queue-item")).to_have_count(4)
            page.locator("#clearButton").click()
            expect(page.locator(".queue-item")).to_have_count(0)
            page.wait_for_function(
                "window.__uiTest.created.every(u => window.__uiTest.revoked.includes(u))"
            )
        finally:
            context.close()

    def run(self) -> int:
        self.record("desktop_laptop_navigation_keyboard_and_revoke", self.layout_navigation)
        self.record(
            "sequential_predictions_presence_history_and_real_downloads",
            self.predictions_exports_history,
        )
        self.record(
            "custom_threshold_reset_and_formula_safe_csv", self.thresholds_reset_formula_csv
        )
        self.record("offline_reconnect_corrupt_error_and_retry", self.errors_retry_offline)
        self.record("file_validation_queue_limit_examples_and_pause", self.limits_examples_pause)
        self.record("upload_size_budget_dedup_safe_names", self.upload_limits_dedup_and_safe_names)
        if self.page_errors:
            self.results.append(
                {
                    "name": "no_uncaught_browser_errors",
                    "status": "failed",
                    "details": self.page_errors,
                }
            )
        else:
            self.results.append({"name": "no_uncaught_browser_errors", "status": "passed"})
        report = {
            "scope": (
                "ISOLATED FRONTEND TEST ONLY: mocked API, "
                "not model accuracy or live-service evidence"
            ),
            "url": self.url,
            "browser": self.browser.version,
            "layout_metrics": self.layout_metrics,
            "results": self.results,
        }
        (self.output / "browser-test-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return int(any(item["status"] != "passed" for item in self.results))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "artifacts/web-redesign-20260914/browser-fixtures"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, channel="chrome")
        try:
            return Suite(browser, args.url, args.output).run()
        finally:
            browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
