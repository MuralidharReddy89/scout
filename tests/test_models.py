from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from scout.models import Citation, Finding, Report

NOW = datetime(2026, 5, 29, 14, 0, 0, tzinfo=UTC)


def _citation(**overrides: object) -> Citation:
    defaults: dict[str, object] = {
        "url": "https://example.com/article",
        "quote": "The capital of France is Paris.",
        "retrieved_at": NOW,
    }
    defaults.update(overrides)
    return Citation.model_validate(defaults)


def _finding(**overrides: object) -> Finding:
    defaults: dict[str, object] = {
        "claim": "Paris is the capital of France.",
        "citations": (_citation(),),
    }
    defaults.update(overrides)
    return Finding.model_validate(defaults)


def test_citation_happy_path() -> None:
    c = _citation()
    assert str(c.url) == "https://example.com/article"
    assert c.quote == "The capital of France is Paris."
    assert c.retrieved_at == NOW


def test_citation_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _citation(retrieved_at=datetime(2026, 5, 29, 14, 0, 0))


def test_citation_rejects_blank_quote() -> None:
    with pytest.raises(ValidationError):
        _citation(quote="   ")


def test_citation_rejects_non_http_url() -> None:
    with pytest.raises(ValidationError):
        _citation(url="ftp://example.com/file.txt")


def test_citation_is_frozen() -> None:
    c = _citation()
    with pytest.raises(ValidationError):
        c.quote = "mutated"


def test_citation_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Citation.model_validate(
            {
                "url": "https://example.com",
                "quote": "x",
                "retrieved_at": NOW,
                "rogue": True,
            }
        )


def test_finding_requires_at_least_one_citation() -> None:
    with pytest.raises(ValidationError):
        Finding.model_validate({"claim": "ungrounded", "citations": ()})


def test_finding_rejects_blank_claim() -> None:
    with pytest.raises(ValidationError):
        _finding(claim="")


def test_report_happy_path() -> None:
    started = NOW
    finished = NOW + timedelta(seconds=5)
    r = Report(
        question="What is the capital of France?",
        findings=(_finding(),),
        model="claude-3-7-sonnet-20250219",
        started_at=started,
        finished_at=finished,
    )
    assert r.question == "What is the capital of France?"
    assert len(r.findings) == 1
    assert r.finished_at >= r.started_at


def test_report_allows_empty_findings_tuple() -> None:
    Report(
        question="Will it rain tomorrow?",
        findings=(),
        model="claude-3-7-sonnet-20250219",
        started_at=NOW,
        finished_at=NOW,
    )


def test_report_rejects_finished_before_started() -> None:
    with pytest.raises(ValidationError, match="finished_at must be >= started_at"):
        Report(
            question="q",
            findings=(),
            model="m",
            started_at=NOW,
            finished_at=NOW - timedelta(seconds=1),
        )


def test_report_rejects_naive_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        Report(
            question="q",
            findings=(),
            model="m",
            started_at=datetime(2026, 5, 29, 14, 0, 0),
            finished_at=NOW,
        )


def test_report_roundtrips_through_json() -> None:
    original = Report(
        question="q",
        findings=(_finding(),),
        model="m",
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
    )
    serialized = original.model_dump_json()
    restored = Report.model_validate_json(serialized)
    assert restored == original
