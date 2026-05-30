from __future__ import annotations

import pytest

from scout.browser.base import BrowserError, Page
from scout.browser.fake import FakeBrowser, RecordedFetch


def _page(url: str = "https://example.com/", text: str = "hi") -> Page:
    return Page(
        url=url,
        final_url=url,
        status_code=200,
        title="t",
        text=text,
        html=f"<html><body>{text}</body></html>",
    )


def test_fetch_returns_registered_page() -> None:
    p = _page()
    b = FakeBrowser({p.url: p})
    assert b.fetch(p.url) == p


def test_add_page_registers_and_overwrites() -> None:
    b = FakeBrowser()
    first = _page(text="first")
    second = _page(text="second")
    b.add_page(first)
    assert b.fetch(first.url).text == "first"
    b.add_page(second)
    assert b.fetch(first.url).text == "second"


def test_fetch_records_url_and_timeout() -> None:
    p = _page()
    b = FakeBrowser({p.url: p})
    b.fetch(p.url)
    b.fetch(p.url, timeout_s=5.5)
    assert b.calls == [
        RecordedFetch(url=p.url, timeout_s=30.0),
        RecordedFetch(url=p.url, timeout_s=5.5),
    ]


def test_unregistered_url_raises_browser_error() -> None:
    b = FakeBrowser()
    with pytest.raises(BrowserError) as exc_info:
        b.fetch("https://missing.example/")
    assert exc_info.value.url == "https://missing.example/"
    # the failed call is still recorded
    assert b.calls == [RecordedFetch(url="https://missing.example/", timeout_s=30.0)]


def test_registered_urls_reports_current_keys() -> None:
    b = FakeBrowser()
    b.add_page(_page(url="https://a.example/"))
    b.add_page(_page(url="https://b.example/"))
    assert set(b.registered_urls) == {"https://a.example/", "https://b.example/"}


def test_constructor_copies_input_dict() -> None:
    p = _page()
    source = {p.url: p}
    b = FakeBrowser(source)
    source.clear()
    assert b.fetch(p.url) == p
