from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version

import pytest
from typer.testing import CliRunner

from scout import __version__, cli
from scout.browser.fake import FakeBrowser
from scout.cli import app
from scout.config import Config
from scout.llm.base import Completion, TextBlock, TokenUsage
from scout.llm.fake import FakeLLMClient

runner = CliRunner()


def _completion(text: str = "done") -> Completion:
    return Completion(
        content=(TextBlock(text=text),),
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        model="fake-model",
    )


@pytest.fixture
def fake_components(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeLLMClient, FakeBrowser]:
    """Replace the live LLM/browser seams with hermetic fakes."""
    llm = FakeLLMClient([_completion("hi")])
    browser = FakeBrowser()
    monkeypatch.setattr(cli, "_build_llm", lambda config: llm)

    @contextmanager
    def _fake_open() -> Iterator[FakeBrowser]:
        yield browser

    monkeypatch.setattr(cli, "_open_browser", _fake_open)
    return llm, browser


def test_version_flag_prints_version_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip() == f"scout {version('scout')}"


def test_version_attribute_matches_distribution_metadata() -> None:
    assert __version__ == version("scout")


def test_no_args_shows_help_and_exits_nonzero() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "Usage:" in result.stdout
    assert "research" in result.stdout


def test_research_missing_api_key_exits_two_with_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["research", "q"])
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.stderr


def test_research_emits_json_report_on_stdout(
    monkeypatch: pytest.MonkeyPatch,
    fake_components: tuple[FakeLLMClient, FakeBrowser],
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    result = runner.invoke(app, ["research", "What is X?"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["question"] == "What is X?"
    assert payload["findings"] == []
    assert payload["summary"] == "hi"
    assert payload["truncated"] is False
    assert payload["over_budget"] is False


def test_research_max_iterations_flag_propagates(
    monkeypatch: pytest.MonkeyPatch,
    fake_components: tuple[FakeLLMClient, FakeBrowser],
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm, _ = fake_components
    result = runner.invoke(app, ["research", "q", "--max-iterations", "1"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(llm.calls) == 1


def test_research_model_flag_overrides_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    seen: dict[str, str] = {}

    def _capture_llm(config: Config) -> FakeLLMClient:
        seen["model"] = config.llm_model
        return FakeLLMClient([_completion("ok")])

    monkeypatch.setattr(cli, "_build_llm", _capture_llm)

    @contextmanager
    def _fake_open() -> Iterator[FakeBrowser]:
        yield FakeBrowser()

    monkeypatch.setattr(cli, "_open_browser", _fake_open)
    result = runner.invoke(app, ["research", "q", "--model", "claude-test"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert seen["model"] == "claude-test"
