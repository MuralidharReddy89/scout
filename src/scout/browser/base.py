from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class BrowserError(RuntimeError):
    """Raised when a browser tool cannot satisfy a fetch request.

    Covers navigation failures, timeouts, non-2xx responses (when the
    caller asked for them to be raised), and tool-launch failures.
    Implementations attach the originating ``url`` on the instance.
    """

    def __init__(self, message: str, *, url: str) -> None:
        super().__init__(message)
        self.url = url


class Page(BaseModel):
    """A snapshot of a single fetched page.

    Frozen so it can be safely passed between the browser tool, the agent
    loop, and citation extraction without anyone mutating it in flight.
    """

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    url: str = Field(min_length=1)
    """The URL the caller requested."""

    final_url: str = Field(min_length=1)
    """The URL after any redirects."""

    status_code: int = Field(ge=100, le=599)
    """HTTP status of the main response."""

    title: str = ""
    """``<title>`` of the loaded document, or empty if absent."""

    text: str
    """Visible text content extracted from ``<body>``."""

    html: str
    """The raw HTML of the loaded document."""


@runtime_checkable
class BrowserTool(Protocol):
    """Provider-agnostic page-fetch interface.

    Synchronous to match :class:`scout.llm.base.LLMClient`; the agent loop
    interacts only through this protocol so that :class:`FakeBrowser` can
    stand in during tests.
    """

    def fetch(self, url: str, *, timeout_s: float = 30.0) -> Page: ...
