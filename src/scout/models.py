from __future__ import annotations

from datetime import datetime
from typing import Annotated, Self

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class _Frozen(BaseModel):
    """Common base: immutable, strict, no extra fields."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class Citation(_Frozen):
    """A single citation backing one claim in a Finding.

    A citation is a verbatim ``quote`` of text retrieved from ``url`` at
    ``retrieved_at``. The quote should appear on the cited page; callers are
    responsible for verification, but downstream consumers may grep for it.
    """

    url: AnyHttpUrl
    quote: NonEmptyStr
    retrieved_at: datetime

    @field_validator("retrieved_at")
    @classmethod
    def _require_tzaware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("retrieved_at must be timezone-aware")
        return v


class Finding(_Frozen):
    """A single claim, backed by one or more Citations.

    The invariant "every claim has at least one citation" is the central
    property of the scout output format and is enforced here at construction.
    """

    claim: NonEmptyStr
    citations: Annotated[tuple[Citation, ...], Field(min_length=1)]


class Report(_Frozen):
    """The full output of one ``scout research <question>`` invocation."""

    question: NonEmptyStr
    findings: tuple[Finding, ...]
    model: NonEmptyStr
    started_at: datetime
    finished_at: datetime
    summary: str = ""
    truncated: bool = False
    over_budget: bool = False

    @field_validator("started_at", "finished_at")
    @classmethod
    def _require_tzaware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("timestamps must be timezone-aware")
        return v

    @model_validator(mode="after")
    def _finished_not_before_started(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must be >= started_at")
        return self
