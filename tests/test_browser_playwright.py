from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from scout.browser.base import BrowserError, BrowserTool
from scout.browser.playwright_browser import PlaywrightBrowser


def _make_browser_with_fake_chromium(
    monkeypatch: pytest.MonkeyPatch,
    response_status: int | None = 200,
    goto_side_effect: BaseException | None = None,
) -> tuple[PlaywrightBrowser, MagicMock, MagicMock]:
    """Wire ``PlaywrightBrowser`` to a fully-stubbed Playwright stack.

    The returned ``page_mock`` is pre-seeded with valid string defaults for
    every attribute :class:`Page` needs; callers override what they care
    about *after* this helper returns.
    """
    page_mock = MagicMock()
    page_mock.url = "https://example.com/"
    page_mock.title.return_value = ""
    page_mock.inner_text.return_value = ""
    page_mock.content.return_value = ""

    response = MagicMock()
    response.status = response_status

    if goto_side_effect is not None:
        page_mock.goto.side_effect = goto_side_effect
    else:
        page_mock.goto.return_value = response if response_status is not None else None

    context = MagicMock()
    context.new_page.return_value = page_mock

    browser_mock = MagicMock()
    browser_mock.new_context.return_value = context

    pw = MagicMock()
    pw.chromium.launch.return_value = browser_mock

    def fake_sync_playwright() -> Any:
        starter = MagicMock()
        starter.start.return_value = pw
        return starter

    monkeypatch.setattr("playwright.sync_api.sync_playwright", fake_sync_playwright, raising=True)

    pb = PlaywrightBrowser()
    return pb, page_mock, context


def test_satisfies_browser_tool_protocol() -> None:
    assert isinstance(PlaywrightBrowser(), BrowserTool)


def test_fetch_rejects_blank_url_and_nonpositive_timeout() -> None:
    pb = PlaywrightBrowser()
    with pytest.raises(BrowserError, match="url"):
        pb.fetch("", timeout_s=10)
    with pytest.raises(BrowserError, match="timeout_s"):
        pb.fetch("https://example.com/", timeout_s=0)


def test_fetch_returns_populated_page(monkeypatch: pytest.MonkeyPatch) -> None:
    pb, page_mock, context = _make_browser_with_fake_chromium(monkeypatch, response_status=200)
    page_mock.url = "https://example.com/final"
    page_mock.title.return_value = "Hello"
    page_mock.inner_text.return_value = "Hello, world."
    page_mock.content.return_value = "<html><body>Hello, world.</body></html>"

    result = pb.fetch("https://example.com/", timeout_s=12)

    assert result.url == "https://example.com/"
    assert result.final_url == "https://example.com/final"
    assert result.status_code == 200
    assert result.title == "Hello"
    assert result.text == "Hello, world."
    assert result.html == "<html><body>Hello, world.</body></html>"

    page_mock.goto.assert_called_once_with("https://example.com/", timeout=12000, wait_until="load")
    context.close.assert_called_once()


def test_fetch_falls_back_to_empty_text_when_body_extraction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pb, page_mock, _ = _make_browser_with_fake_chromium(monkeypatch)
    page_mock.inner_text.side_effect = PlaywrightError("no body")
    page_mock.content.return_value = "<html></html>"

    result = pb.fetch("https://example.com/")
    assert result.text == ""
    assert result.html == "<html></html>"


def test_navigation_timeout_raises_browser_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pb, _, context = _make_browser_with_fake_chromium(
        monkeypatch, goto_side_effect=PlaywrightTimeoutError("slow")
    )
    with pytest.raises(BrowserError, match="timeout"):
        pb.fetch("https://example.com/", timeout_s=2)
    context.close.assert_called_once()


def test_navigation_error_raises_browser_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pb, _, _ = _make_browser_with_fake_chromium(
        monkeypatch, goto_side_effect=PlaywrightError("dns")
    )
    with pytest.raises(BrowserError, match="navigation failed"):
        pb.fetch("https://example.com/")


def test_none_response_raises_browser_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pb, _, _ = _make_browser_with_fake_chromium(monkeypatch, response_status=None)
    with pytest.raises(BrowserError, match="no response"):
        pb.fetch("https://example.com/")


def test_close_is_idempotent_and_releases_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    pb, _, _ = _make_browser_with_fake_chromium(monkeypatch)
    pb.fetch("https://example.com/")  # forces start
    started: object = pb._browser
    assert started is not None
    pb.close()
    after_close: object = pb._browser
    assert after_close is None
    pb.close()  # second call must not raise


def test_context_manager_closes_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    pb, _, _ = _make_browser_with_fake_chromium(monkeypatch)
    with pb as opened:
        opened.fetch("https://example.com/")
        opened_browser: object = opened._browser
        assert opened_browser is not None
    final_browser: object = pb._browser
    assert final_browser is None


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("SCOUT_RUN_BROWSER_INTEGRATION") != "1",
    reason="SCOUT_RUN_BROWSER_INTEGRATION!=1; skipping live chromium fetch",
)
def test_live_fetch_against_example_com() -> None:
    with PlaywrightBrowser() as pb:
        page = pb.fetch("https://example.com/", timeout_s=30)
    assert page.status_code == 200
    assert "Example Domain" in page.title
    assert "Example Domain" in page.text
