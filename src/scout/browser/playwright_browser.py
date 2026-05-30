from __future__ import annotations

from contextlib import suppress
from types import TracebackType
from typing import TYPE_CHECKING

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from scout.browser.base import BrowserError, Page

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Playwright


class PlaywrightBrowser:
    """``BrowserTool`` backed by a headless Chromium via Playwright sync API.

    A single Chromium process is launched on first use and reused across
    fetches. Each ``fetch`` spawns a fresh ``BrowserContext`` so cookies
    and storage from one call don't leak into another. Use as a context
    manager (or call :meth:`close` explicitly) to release the browser.
    """

    def __init__(
        self,
        *,
        user_agent: str = "scout-research-agent/0.1",
        headless: bool = True,
    ) -> None:
        self._user_agent = user_agent
        self._headless = headless
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    def __enter__(self) -> PlaywrightBrowser:
        self._ensure_started()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Tear down the browser and Playwright driver. Idempotent."""
        if self._browser is not None:
            with suppress(PlaywrightError):
                self._browser.close()
            self._browser = None
        if self._playwright is not None:
            with suppress(PlaywrightError):
                self._playwright.stop()
            self._playwright = None

    def fetch(self, url: str, *, timeout_s: float = 30.0) -> Page:
        if not url:
            raise BrowserError("url must be a non-empty string", url=url)
        if timeout_s <= 0:
            raise BrowserError("timeout_s must be positive", url=url)

        browser = self._ensure_started()
        context = browser.new_context(user_agent=self._user_agent)
        try:
            page = context.new_page()
            try:
                response = page.goto(url, timeout=timeout_s * 1000, wait_until="load")
            except PlaywrightTimeoutError as exc:
                raise BrowserError(f"navigation timeout after {timeout_s}s", url=url) from exc
            except PlaywrightError as exc:
                raise BrowserError(f"navigation failed: {exc}", url=url) from exc

            if response is None:
                raise BrowserError("navigation produced no response", url=url)

            try:
                body_text = page.inner_text("body")
            except PlaywrightError:
                body_text = ""

            return Page(
                url=url,
                final_url=page.url,
                status_code=response.status,
                title=page.title(),
                text=body_text,
                html=page.content(),
            )
        finally:
            with suppress(PlaywrightError):
                context.close()

    def _ensure_started(self) -> Browser:
        if self._browser is not None:
            return self._browser
        from playwright.sync_api import sync_playwright

        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self._headless)
        except PlaywrightError as exc:
            self.close()
            raise BrowserError(f"failed to launch chromium: {exc}", url="") from exc
        return self._browser
