from __future__ import annotations

from importlib.metadata import version

from typer.testing import CliRunner

from scout import __version__
from scout.cli import app

runner = CliRunner()


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


def test_research_is_not_implemented_yet() -> None:
    result = runner.invoke(app, ["research", "what is the capital of France?"])
    assert result.exit_code == 2
    assert "not implemented" in result.stderr.lower()
