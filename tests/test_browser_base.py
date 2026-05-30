from __future__ import annotations

import pytest
from pydantic import ValidationError

from scout.browser.base import BrowserError, BrowserTool, Page
from scout.browser.fake import FakeBrowser


def _page(
    *,
    url: str = "https://example.com/a",
    final_url: str = "https://example.com/a",
    status_code: int = 200,
    title: str = "Hello",
    text: str = "Hello, world.",
    html: str = "<html><body>Hello, world.</body></html>",
) -> Page:
    return Page(
        url=url,
        final_url=final_url,
        status_code=status_code,
        title=title,
        text=text,
        html=html,
    )


def test_page_round_trip_via_model_dump() -> None:
    p = _page()
    assert Page(**p.model_dump()) == p


def test_page_is_frozen() -> None:
    p = _page()
    with pytest.raises(ValidationError):
        p.url = "https://other.example/"


def test_page_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Page(
            url="https://example.com/",
            final_url="https://example.com/",
            status_code=200,
            text="x",
            html="<p>x</p>",
            extra="nope",  # type: ignore[call-arg]
        )


def test_page_rejects_blank_url() -> None:
    with pytest.raises(ValidationError):
        _page(url="")


def test_page_status_code_bounds() -> None:
    for bad in (99, 600, -1):
        with pytest.raises(ValidationError):
            _page(status_code=bad)
    for ok in (100, 200, 404, 599):
        assert _page(status_code=ok).status_code == ok


def test_page_title_defaults_to_empty() -> None:
    p = Page(
        url="https://example.com/",
        final_url="https://example.com/",
        status_code=200,
        text="",
        html="",
    )
    assert p.title == ""


def test_browser_error_carries_url() -> None:
    err = BrowserError("boom", url="https://example.com/x")
    assert err.url == "https://example.com/x"
    assert str(err) == "boom"


def test_fake_browser_satisfies_browser_tool_protocol() -> None:
    assert isinstance(FakeBrowser(), BrowserTool)
