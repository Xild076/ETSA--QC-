import os
import sys
import re
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

APP_URL = os.getenv("APP_URL", "https://etsa-survey.streamlit.app/").strip()
SLEEP_PAGE_TEXT = os.getenv("SLEEP_PAGE_TEXT", "app is asleep")
MINIMUM_CONTENT_LENGTH = int(os.getenv("MINIMUM_CONTENT_LENGTH", "1000"))
KEEP_OPEN_MS = int(os.getenv("KEEP_OPEN_MS", "8000"))

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)

def log(msg: str):
    print(msg, flush=True)
    with open("debug.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def save_artifacts(page, always=False):
    try:
        html = page.content()
        Path("response.html").write_text(html, encoding="utf-8")
    except Exception as e:
        log(f"WARN: Failed to save response.html: {e}")
    # Screenshot is more expensive; take it only on failure unless always=True
    if always:
        try:
            page.screenshot(path="screenshot.png", full_page=False)
        except Exception as e:
            log(f"WARN: Failed to save screenshot.png: {e}")

def main() -> int:
    log(f"Keep-alive: {APP_URL}")
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
            ignore_https_errors=True,
            viewport={"width": 1200, "height": 800},
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://google.com",
            },
        )

        # Block heavy resources to speed up load
        def block_heavy(route):
            if route.request.resource_type in ("image", "media", "font"):
                return route.abort()
            return route.continue_()
        context.route("**/*", block_heavy)

        ws_urls = []
        context.on("websocket", lambda ws: (ws_urls.append(ws.url), log(f"WebSocket opened: {ws.url}")))

        page = context.new_page()

        try:
            log("Navigating...")
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)

            # Try to detect Streamlit DOM quickly
            selectors = [
                '[data-testid="stAppViewContainer"]',
                'section.main',
                '#root',
            ]
            found = False
            for sel in selectors:
                try:
                    page.wait_for_selector(sel, timeout=8_000)
                    log(f"Detected Streamlit container: {sel}")
                    found = True
                    break
                except PWTimeout:
                    continue
            if not found:
                log("WARN: Streamlit container not confirmed; continuing.")

            # Keep open briefly to ensure the session + WS are established
            keep_open_ms = max(KEEP_OPEN_MS, 5000)
            page.wait_for_timeout(keep_open_ms)

            # Save artifacts and validate
            save_artifacts(page)
            try:
                html = Path("response.html").read_text(encoding="utf-8")
            except Exception:
                html = page.content()

            content_length = len(html.encode("utf-8", errors="ignore"))
            log(f"Content length: {content_length} bytes")
            if content_length < MINIMUM_CONTENT_LENGTH:
                log(f"CRITICAL: Content below minimum ({MINIMUM_CONTENT_LENGTH}).")
                save_artifacts(page, always=True)
                return 47

            lower_html = html.lower()
            challenge_patterns = [
                r"checking your browser",
                r"just a moment",
                r"cf-browser-verification",
                r"<meta[^>]*http-equiv=['\"]refresh['\"]",
                r"window\.location",
            ]
            if SLEEP_PAGE_TEXT.lower() in lower_html:
                log("CRITICAL: Sleep message detected.")
                save_artifacts(page, always=True)
                return 47
            for pat in challenge_patterns:
                if re.search(pat, lower_html):
                    log(f"CRITICAL: Challenge/redirect pattern detected: {pat}")
                    save_artifacts(page, always=True)
                    return 47

            if ws_urls:
                log(f"✅ Session established (WebSockets observed: {len(ws_urls)}).")
            else:
                log("ℹ️ No WebSocket observed; session may still be active.")

            log("✅ Completed keep-alive.")
            return 0

        except PWTimeout:
            save_artifacts(page, always=True)
            log("CRITICAL: Navigation timeout.")
            return 47
        except Exception as e:
            save_artifacts(page, always=True)
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
