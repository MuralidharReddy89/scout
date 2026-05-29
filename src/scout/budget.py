from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field


class BudgetExceeded(RuntimeError):
    """Raised when consumption would push usage past a configured cap."""

    def __init__(self, axis: str, used: float, limit: float) -> None:
        super().__init__(f"budget exceeded on '{axis}': used={used}, limit={limit}")
        self.axis = axis
        self.used = used
        self.limit = limit


class BudgetLimits(BaseModel):
    """Immutable per-axis caps. ``None`` means no cap on that axis."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    max_tokens: int | None = Field(default=None, ge=0)
    max_usd: float | None = Field(default=None, ge=0)
    max_pages: int | None = Field(default=None, ge=0)
    max_wall_seconds: float | None = Field(default=None, ge=0)


class BudgetUsage(BaseModel):
    """Snapshot of accumulated consumption."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

    tokens: int = 0
    usd: float = 0.0
    pages: int = 0
    wall_seconds: float = 0.0


class Budget:
    """Mutable budget tracker.

    Holds a set of caps (``BudgetLimits``) and an accumulator of usage.
    Each ``consume_*`` call raises ``BudgetExceeded`` if it would push
    the corresponding axis past its cap. The wall clock starts on
    construction; ``check_wall_clock()`` raises if the limit has elapsed.
    """

    def __init__(
        self,
        limits: BudgetLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limits = limits
        self._clock = clock
        self._started_at = clock()
        self._tokens = 0
        self._usd = 0.0
        self._pages = 0

    @property
    def limits(self) -> BudgetLimits:
        return self._limits

    @property
    def usage(self) -> BudgetUsage:
        return BudgetUsage(
            tokens=self._tokens,
            usd=self._usd,
            pages=self._pages,
            wall_seconds=self._wall_seconds(),
        )

    def _wall_seconds(self) -> float:
        return self._clock() - self._started_at

    def _check(self, axis: str, used: float, limit: float | None) -> None:
        if limit is not None and used > limit:
            raise BudgetExceeded(axis, used, limit)

    def consume_tokens(self, n: int) -> None:
        if n < 0:
            raise ValueError("token consumption must be non-negative")
        self._tokens += n
        self._check("tokens", self._tokens, self._limits.max_tokens)

    def consume_usd(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("usd consumption must be non-negative")
        self._usd += amount
        self._check("usd", self._usd, self._limits.max_usd)

    def consume_page(self, n: int = 1) -> None:
        if n < 0:
            raise ValueError("page consumption must be non-negative")
        self._pages += n
        self._check("pages", self._pages, self._limits.max_pages)

    def check_wall_clock(self) -> None:
        self._check("wall_seconds", self._wall_seconds(), self._limits.max_wall_seconds)
