from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:18080"
EXAMPLE = (
    ROOT
    / "datasets"
    / "mikania_detection_v2_0_1"
    / "images"
    / "test_internal"
    / "IMG_20251118_165528.jpg"
)
DESKTOP_SCREENSHOT = ROOT / "artifacts" / "ui" / "mikania-detection-page.png"
MOBILE_SCREENSHOT = ROOT / "artifacts" / "ui" / "mikania-detection-mobile-v2_1.png"


def main() -> None:
    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on(
            "console",
            lambda message: errors.append(message.text) if message.type == "error" else None,
        )
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        assert page.locator("#modelState").inner_text().strip().endswith("识别模型已就绪")
        assert page.locator('input[name="scanMode"]:checked').get_attribute("value") == "precise"
        assert page.locator("#thresholdValue").inner_text() == "15%"

        page.locator("#fileInput").set_input_files(str(EXAMPLE))
        page.locator("#detectButton").click()
        page.locator("#result:not([hidden])").wait_for(timeout=30_000)
        assert page.locator("#resultMode").inner_text() == "精细"
        assert int(page.locator("#resultCount").inner_text()) >= 1
        page.screenshot(path=str(DESKTOP_SCREENSHOT), full_page=True)

        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(BASE_URL)
        mobile.wait_for_load_state("networkidle")
        assert mobile.locator('input[name="scanMode"]:checked').get_attribute("value") == "precise"
        mobile.screenshot(path=str(MOBILE_SCREENSHOT), full_page=True)
        browser.close()

    print(
        json.dumps(
            {
                "status": "passed",
                "scan_mode": "precise",
                "threshold": "15%",
                "console_errors": errors,
                "desktop_screenshot": str(DESKTOP_SCREENSHOT),
                "mobile_screenshot": str(MOBILE_SCREENSHOT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
