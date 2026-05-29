from __future__ import annotations

import pytest
from pydantic import ValidationError

from scout.budget import Budget, BudgetExceeded, BudgetLimits, BudgetUsage


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_limits_default_to_none_on_every_axis() -> None:
    lim = BudgetLimits()
    assert lim.max_tokens is None
    assert lim.max_usd is None
    assert lim.max_pages is None
    assert lim.max_wall_seconds is None


def test_limits_reject_negative_values() -> None:
    with pytest.raises(ValidationError):
        BudgetLimits(max_tokens=-1)
    with pytest.raises(ValidationError):
        BudgetLimits(max_usd=-0.01)


def test_unbounded_budget_never_raises() -> None:
    b = Budget(BudgetLimits(), clock=FakeClock())
    b.consume_tokens(10**9)
    b.consume_usd(10**6)
    b.consume_page(10**6)
    b.check_wall_clock()
    assert b.usage.tokens == 10**9


def test_token_cap_raises_with_axis_info() -> None:
    b = Budget(BudgetLimits(max_tokens=100), clock=FakeClock())
    b.consume_tokens(60)
    with pytest.raises(BudgetExceeded) as exc:
        b.consume_tokens(50)
    assert exc.value.axis == "tokens"
    assert exc.value.limit == 100
    assert exc.value.used == 110


def test_token_cap_allows_exact_match() -> None:
    b = Budget(BudgetLimits(max_tokens=100), clock=FakeClock())
    b.consume_tokens(100)
    assert b.usage.tokens == 100


def test_usd_cap_raises() -> None:
    b = Budget(BudgetLimits(max_usd=1.0), clock=FakeClock())
    b.consume_usd(0.5)
    with pytest.raises(BudgetExceeded) as exc:
        b.consume_usd(0.6)
    assert exc.value.axis == "usd"


def test_pages_cap_raises() -> None:
    b = Budget(BudgetLimits(max_pages=2), clock=FakeClock())
    b.consume_page()
    b.consume_page()
    with pytest.raises(BudgetExceeded) as exc:
        b.consume_page()
    assert exc.value.axis == "pages"
    assert exc.value.used == 3


def test_wall_clock_cap_uses_injected_clock() -> None:
    clk = FakeClock(t=100.0)
    b = Budget(BudgetLimits(max_wall_seconds=5.0), clock=clk)
    clk.advance(4.0)
    b.check_wall_clock()
    clk.advance(2.0)
    with pytest.raises(BudgetExceeded) as exc:
        b.check_wall_clock()
    assert exc.value.axis == "wall_seconds"
    assert exc.value.limit == 5.0


def test_consume_rejects_negative_amounts() -> None:
    b = Budget(BudgetLimits(), clock=FakeClock())
    with pytest.raises(ValueError, match="non-negative"):
        b.consume_tokens(-1)
    with pytest.raises(ValueError, match="non-negative"):
        b.consume_usd(-0.01)
    with pytest.raises(ValueError, match="non-negative"):
        b.consume_page(-1)


def test_usage_snapshot_is_immutable() -> None:
    b = Budget(BudgetLimits(), clock=FakeClock())
    b.consume_tokens(5)
    snap = b.usage
    assert isinstance(snap, BudgetUsage)
    with pytest.raises(ValidationError):
        snap.tokens = 999


def test_limits_property_exposes_original_limits() -> None:
    lim = BudgetLimits(max_tokens=42)
    b = Budget(lim, clock=FakeClock())
    assert b.limits is lim
