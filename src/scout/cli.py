from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from scout.browser.base import BrowserTool
    from scout.config import Config
    from scout.llm.base import LLMClient

app = typer.Typer(
    name="scout",
    help="LLM-driven web research agent.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if not value:
        return
    try:
        v = version("scout")
    except PackageNotFoundError:
        v = "0.0.0+unknown"
    typer.echo(f"scout {v}")
    raise typer.Exit()


@app.callback()
def _root(
    _version_flag: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the scout version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Root callback. Subcommands hang off this."""


@app.command()
def research(
    question: Annotated[str, typer.Argument(help="The research question to answer.")],
    model: Annotated[
        str | None,
        typer.Option("--model", help="Override SCOUT_LLM_MODEL for this run."),
    ] = None,
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", min=1, help="Maximum agent loop iterations."),
    ] = 12,
) -> None:
    """Run a research task and print a JSON Report to stdout."""
    from scout.agent import Researcher
    from scout.budget import Budget
    from scout.config import Config

    config = Config.from_env()
    if not config.anthropic_api_key:
        typer.echo("error: ANTHROPIC_API_KEY is not set in the environment.", err=True)
        raise typer.Exit(code=2)
    if model:
        config = config.model_copy(update={"llm_model": model})

    llm = _build_llm(config)
    budget = Budget(config.budget_limits)
    with _open_browser() as browser:
        researcher = Researcher(
            llm=llm,
            browser=browser,
            budget=budget,
            max_iterations=max_iterations,
        )
        report = researcher.research(question)
    typer.echo(report.model_dump_json(indent=2))


def _build_llm(config: Config) -> LLMClient:
    """Construct the live ``LLMClient``. Tests monkeypatch this seam."""
    from scout.llm.anthropic_client import AnthropicClient

    assert config.anthropic_api_key is not None
    return AnthropicClient(api_key=config.anthropic_api_key, model=config.llm_model)


@contextmanager
def _open_browser() -> Iterator[BrowserTool]:
    """Open the live ``BrowserTool`` as a context manager. Tests monkeypatch this seam."""
    from scout.browser.playwright_browser import PlaywrightBrowser

    with PlaywrightBrowser() as browser:
        yield browser


def main() -> None:
    app()


if __name__ == "__main__":
    main()
