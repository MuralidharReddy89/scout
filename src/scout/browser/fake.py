from __future__ import annotations

from dataclasses import dataclass

from scout.browser.base import BrowserError, Page


@dataclass(frozen=True)
class RecordedFetch:
    """Snapshot of one ``fetch`` invocation, for assertions in tests."""

    url: str
    timeout_s: float


class FakeBrowser:
    """Deterministic, in-memory stand-in for the ``BrowserTool`` protocol.

    Register pages via :meth:`add_page` (or pre-seed in the constructor);
    every ``fetch`` returns the page registered under that exact URL and
    records the call. URLs that have not been registered raise
    :class:`BrowserError`, mirroring how :class:`PlaywrightBrowser`
    surfaces navigation failures.
    """

    def __init__(self, pages: dict[str, Page] | None = None) -> None:
        self._pages: dict[str, Page] = dict(pages) if pages else {}
        self.calls: list[RecordedFetch] = []

    def add_page(self, page: Page) -> None:
        """Register ``page`` under its ``url`` (replaces any prior entry)."""
        self._pages[page.url] = page

    def fetch(self, url: str, *, timeout_s: float = 30.0) -> Page:
        self.calls.append(RecordedFetch(url=url, timeout_s=timeout_s))
        try:
            return self._pages[url]
        except KeyError as exc:
            raise BrowserError(f"no page registered for url: {url!r}", url=url) from exc

    @property
    def registered_urls(self) -> tuple[str, ...]:
        return tuple(self._pages)
