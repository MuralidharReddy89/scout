from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from scout.budget import BudgetLimits

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_PAGES = 20
DEFAULT_MAX_WALL_SECONDS = 120.0


class Config(BaseModel):
    """Top-level runtime configuration.

    Frozen so tests can pass instances around without worrying about
    accidental mutation by intermediate components.
    """

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    anthropic_api_key: str | None = None
    llm_model: str = Field(default=DEFAULT_MODEL, min_length=1)
    budget_limits: BudgetLimits = Field(
        default_factory=lambda: BudgetLimits(
            max_pages=DEFAULT_MAX_PAGES,
            max_wall_seconds=DEFAULT_MAX_WALL_SECONDS,
        )
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Self:
        """Build a Config by reading well-known env vars.

        Recognized variables (all optional, missing => default):
        - ``ANTHROPIC_API_KEY``     : API key for the Anthropic adapter
        - ``SCOUT_LLM_MODEL``       : LLM model identifier
        - ``SCOUT_MAX_TOKENS``      : int cap on total tokens
        - ``SCOUT_MAX_USD``         : float cap on total spend
        - ``SCOUT_MAX_PAGES``       : int cap on pages visited
        - ``SCOUT_MAX_WALL_SECONDS``: float cap on wall-clock duration
        """
        e: Mapping[str, str] = env if env is not None else os.environ
        limits = BudgetLimits(
            max_tokens=_int(e, "SCOUT_MAX_TOKENS"),
            max_usd=_float(e, "SCOUT_MAX_USD"),
            max_pages=_int(e, "SCOUT_MAX_PAGES", default=DEFAULT_MAX_PAGES),
            max_wall_seconds=_float(e, "SCOUT_MAX_WALL_SECONDS", default=DEFAULT_MAX_WALL_SECONDS),
        )
        raw_key = e.get("ANTHROPIC_API_KEY")
        api_key = raw_key.strip() if raw_key else None
        return cls(
            anthropic_api_key=api_key or None,
            llm_model=e.get("SCOUT_LLM_MODEL") or DEFAULT_MODEL,
            budget_limits=limits,
        )


def _int(env: Mapping[str, str], key: str, *, default: int | None = None) -> int | None:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"env var {key}={raw!r} is not a valid int") from exc


def _float(env: Mapping[str, str], key: str, *, default: float | None = None) -> float | None:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"env var {key}={raw!r} is not a valid float") from exc
