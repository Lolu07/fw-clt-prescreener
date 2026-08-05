"""Open the deployed Streamlit app in a real browser so Community Cloud sees a
live session, and click "get this app back up" if it has gone to sleep.

A plain HTTP GET is not enough. Community Cloud answers requests to the app URL
from its router with the same SPA shell whether the app is awake or asleep, and
it decides an app is idle based on real app sessions (the websocket to
/_stcore/stream) that only open once JavaScript runs. The user's app is also
rendered inside an /~/+/ iframe, so every check below runs against that frame.
"""

import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "https://fw-clt-prescreener-platform.streamlit.app/")
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
# A string that only appears once the app script has actually executed, so we
# distinguish "Streamlit frontend mounted" from "the app really ran".
EXPECT_TEXT = os.environ.get("EXPECT_TEXT", "Pre-Screener")

APP_FRAME = 'iframe[src*="/~/+/"]'
APP_READY = '[data-testid="stAppViewContainer"]'
# Match on the button text: Streamlit has reworded the copy around it over time.
WAKE_BUTTON = 'button:has-text("get this app back up")'
# Cold starts have to reinstall dependencies and re-run the whole script.
WAKE_TIMEOUT_MS = 5 * 60 * 1000


def log(msg):
    print(msg, flush=True)


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        # Desktop viewport: the mobile layout puts the wake button elsewhere.
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        def fail(reason):
            page.screenshot(path=f"{ARTIFACT_DIR}/failure.png", full_page=True)
            log(f"FAILED: {reason}")
            return 1

        log(f"opening {APP_URL}")
        # The app URL 303-redirects via share.streamlit.io to set a session
        # cookie; the browser follows that handshake by itself.
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=90_000)

        try:
            page.wait_for_selector(WAKE_BUTTON, timeout=20_000)
        except PlaywrightTimeout:
            log("no sleep screen - app was already awake")
        else:
            log("app was asleep - clicking wake button")
            page.click(WAKE_BUTTON)

        try:
            page.frame_locator(APP_FRAME).locator(APP_READY).wait_for(
                timeout=WAKE_TIMEOUT_MS
            )
        except PlaywrightTimeout:
            return fail(f"app frame never rendered {APP_READY}")

        # Hold the session open so it reads as real traffic rather than a
        # connect/disconnect blip, and give the script time to finish rendering.
        page.wait_for_timeout(20_000)

        frame = next((f for f in page.frames if "/~/+/" in (f.url or "")), None)
        if frame is None:
            return fail("app frame disappeared after load")

        body = frame.inner_text("body")
        if EXPECT_TEXT not in body:
            return fail(f"{EXPECT_TEXT!r} not found in rendered app ({len(body)} chars)")

        page.screenshot(path=f"{ARTIFACT_DIR}/app.png")
        log(f"OK: app is awake and rendered ({len(body)} chars of content)")

        context.close()
        browser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
