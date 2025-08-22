import os
import sys
import time
import re
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

APP_URL = os.getenv("APP_URL", "https://etsa-survey.streamlit.app/").strip()
SLEEP_PAGE_TEXT = os.getenv("SLEEP_PAGE_TEXT", "app is asleep")
MINIMUM_CONTENT_LENGTH = int(os.getenv("MINIMUM_CONTENT_LENGTH", "1000"))
KEEP_OPEN_MS = int(os.getenv("KEEP_OPEN_MS", "20000"))

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)

def log(msg: str):
    print(msg, flush=True)
    with open("debug.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def save_artifacts(page):
    # Save the HTML and a screenshot for debugging
    try:
        html = page.content()
        Path("response.html").write_text(html, encoding="utf-8")
    except Exception as e:
        log(f"WARN: Failed to save response.html: {e}")
    try:
        page.screenshot(path="screenshot.png", full_page=True)
    except Exception as e:
        log(f"WARN: Failed to save screenshot.png: {e}")

def main() -> int:
    log(f"Starting headless keep-alive for {APP_URL}")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
            ],
        )
        context = browser.new_context(
            user_agent=UA,
            ignore_https_errors=True,  # mirror 'requests' behavior in your example
            viewport={"width": 1366, "height": 900},
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://google.com",
            },
        )

        ws_urls = []
        def on_ws(ws):
            ws_urls.append(ws.url)
            log(f"WebSocket opened: {ws.url}")
        context.on("websocket", on_ws)

        page = context.new_page()

        try:
            log("Navigating to app...")
            # Wait for DOM load; networkidle may hang depending on app behavior
            page.goto(APP_URL, wait_until="load", timeout=90_000)

            # Common Streamlit containers to wait for
            selectors = [
                '[data-testid="stAppViewContainer"]',
                'section.main',  # older Streamlit markup
                '#root',
            ]
            found = False
            for sel in selectors:
                try:
                    page.wait_for_selector(sel, timeout=20_000)
                    log(f"Detected Streamlit container: {sel}")
                    found = True
                    break
                except PWTimeout:
                    continue

            if not found:
                log("WARN: Could not detect a known Streamlit container; continuing anyway.")

            # Give the frontend time to establish a session/WebSocket
            keep_open_ms = max(KEEP_OPEN_MS, 5000)
            log(f"Keeping page open for {keep_open_ms} ms to establish/maintain session...")
            page.wait_for_timeout(keep_open_ms)

            # Optional: touch health endpoint via page (same-origin)
            try:
                page.evaluate("() => fetch('/_stcore/health').catch(() => {})")
                page.wait_for_timeout(1000)
            except Exception as e:
                log(f"WARN: Health ping failed: {e}")

            # Capture artifacts and validate content
            save_artifacts(page)
            html = ""
            try:
                html = Path("response.html").read_text(encoding="utf-8")
            except Exception:
                html = page.content()

            content_length = len(html.encode("utf-8", errors="ignore"))
            log(f"Content length: {content_length} bytes")
            if content_length < MINIMUM_CONTENT_LENGTH:
                log(f"CRITICAL: Content length below minimum threshold ({MINIMUM_CONTENT_LENGTH}).")
                return 47

            # Detect "sleep" message or bot-challenge text
            lower_html = html.lower()
            challenge_patterns = [
                r"checking your browser",
                r"just a moment",
                r"cf-browser-verification",
                r"<meta[^>]*http-equiv=['\"]refresh['\"]",
                r"window\.location",
            ]
            if SLEEP_PAGE_TEXT.lower() in lower_html:
                log("CRITICAL: Sleep message detected in page.")
                return 47
            for pat in challenge_patterns:
                if re.search(pat, lower_html):
                    log(f"CRITICAL: Detected challenge/redirect pattern: {pat}")
                    return 47

            # If we saw a WebSocket, it's a strong indicator the session was actually established.
            if ws_urls:
                log(f"✅ Success: Established Streamlit session (WebSockets observed: {len(ws_urls)}).")
            else:
                log("ℹ️ Note: No WebSocket observed. Session may still be initialized, but this is less certain.")

            log("✅ Keep-alive step completed successfully.")
            return 0

        except PWTimeout:
            save_artifacts(page)
            log("CRITICAL: Navigation timeout.")
            return 47
        except Exception as e:
            save_artifacts(page)
            log(f"CRITICAL: Unexpected error: {e}")
            return 47
        finally:
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass

if __name__ == "__main__":
    sys.exit(main())
