from __future__ import annotations

from datetime import UTC, datetime
from itertools import count
from typing import Any

import pytest

from scout.agent import FETCH_URL_TOOL, RECORD_FINDING_TOOL, Researcher
from scout.browser.base import Page
from scout.browser.fake import FakeBrowser
from scout.budget import Budget, BudgetLimits
from scout.llm.base import Completion, TextBlock, TokenUsage, ToolUseBlock
from scout.llm.fake import FakeLLMClient

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
PAGE_TEXT = "Paris is the capital of France. The Eiffel Tower stands in Paris."
PAGE_URL = "https://example.com/paris"


def _page(url: str = PAGE_URL, text: str = PAGE_TEXT) -> Page:
    return Page(
        url=url,
        final_url=url,
        status_code=200,
        title="Paris",
        text=text,
        html=f"<html><body>{text}</body></html>",
    )


def _completion(*blocks: TextBlock | ToolUseBlock, stop: str = "tool_use") -> Completion:
    return Completion(
        content=tuple(blocks),
        stop_reason="end_turn" if stop == "end_turn" else "tool_use",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        model="fake-model",
    )


def _fetch_call(call_id: str, url: str = PAGE_URL) -> ToolUseBlock:
    return ToolUseBlock(id=call_id, name=FETCH_URL_TOOL.name, input={"url": url})


def _record_call(call_id: str, **input_overrides: Any) -> ToolUseBlock:
    args: dict[str, Any] = {
        "claim": "Paris is the capital of France.",
        "source_url": PAGE_URL,
        "quote": "Paris is the capital of France.",
    }
    args.update(input_overrides)
    return ToolUseBlock(id=call_id, name=RECORD_FINDING_TOOL.name, input=args)


def _budget(**limits: Any) -> Budget:
    return Budget(BudgetLimits(**limits), clock=lambda: 0.0)


def _researcher(
    llm: FakeLLMClient,
    browser: FakeBrowser,
    budget: Budget | None = None,
    **kwargs: Any,
) -> Researcher:
    ticks = count()
    return Researcher(
        llm=llm,
        browser=browser,
        budget=budget or _budget(),
        clock=lambda: EPOCH.replace(second=next(ticks) % 60),
        **kwargs,
    )


def test_happy_path_fetch_then_record_then_stop() -> None:
    llm = FakeLLMClient(
        [
            _completion(_fetch_call("c1")),
            _completion(_record_call("c2")),
            _completion(TextBlock(text="Done."), stop="end_turn"),
        ]
    )
    browser = FakeBrowser({PAGE_URL: _page()})
    report = _researcher(llm, browser).research("What is the capital of France?")
    assert len(report.findings) == 1
    assert report.findings[0].claim == "Paris is the capital of France."
    assert str(report.findings[0].citations[0].url) == PAGE_URL
    assert report.summary == "Done."
    assert report.truncated is False
    assert report.over_budget is False
    assert report.model == "fake-model"
    assert len(browser.calls) == 1


def test_max_iterations_marks_report_truncated() -> None:
    llm = FakeLLMClient([_completion(_fetch_call(f"c{i}")) for i in range(5)])
    browser = FakeBrowser({PAGE_URL: _page()})
    report = _researcher(llm, browser, max_iterations=3).research("q")
    assert report.truncated is True
    assert report.over_budget is False
    assert len(llm.calls) == 3


def test_wall_clock_budget_stops_before_first_call() -> None:
    tick = iter([0.0, 99.0, 99.0, 99.0])
    budget = Budget(BudgetLimits(max_wall_seconds=1.0), clock=lambda: next(tick))
    llm = FakeLLMClient([_completion(TextBlock(text="x"), stop="end_turn")])
    report = _researcher(llm, FakeBrowser(), budget=budget).research("q")
    assert report.over_budget is True
    assert llm.calls == []


def test_token_budget_stops_after_consuming() -> None:
    llm = FakeLLMClient([_completion(_fetch_call("c1"))])
    report = _researcher(
        llm, FakeBrowser({PAGE_URL: _page()}), budget=_budget(max_tokens=5)
    ).research("q")
    assert report.over_budget is True
    assert report.findings == ()


def test_browser_error_is_reported_as_tool_result_and_loop_continues() -> None:
    llm = FakeLLMClient(
        [
            _fetch := _completion(_fetch_call("c1", url="https://missing.example/x")),
            _completion(TextBlock(text="giving up"), stop="end_turn"),
        ]
    )
    report = _researcher(llm, FakeBrowser()).research("q")
    assert report.summary == "giving up"
    assert report.findings == ()
    assert len(llm.calls) == 2


def test_record_finding_rejects_unfetched_source_url() -> None:
    llm = FakeLLMClient(
        [
            _completion(_record_call("c1")),
            _completion(TextBlock(text="ok"), stop="end_turn"),
        ]
    )
    report = _researcher(llm, FakeBrowser()).research("q")
    assert report.findings == ()


def test_record_finding_rejects_quote_not_on_page() -> None:
    llm = FakeLLMClient(
        [
            _completion(_fetch_call("c1")),
            _completion(_record_call("c2", quote="invented text not on page")),
            _completion(TextBlock(text="ok"), stop="end_turn"),
        ]
    )
    report = _researcher(llm, FakeBrowser({PAGE_URL: _page()})).research("q")
    assert report.findings == ()


def test_invalid_tool_args_reported_and_loop_continues() -> None:
    llm = FakeLLMClient(
        [
            _completion(ToolUseBlock(id="c1", name=FETCH_URL_TOOL.name, input={})),
            _completion(TextBlock(text="bye"), stop="end_turn"),
        ]
    )
    report = _researcher(llm, FakeBrowser()).research("q")
    assert report.summary == "bye"
    assert len(llm.calls) == 2


def test_research_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="question"):
        _researcher(FakeLLMClient(), FakeBrowser()).research("   ")
