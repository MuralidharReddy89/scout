from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Annotated

import typer

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
    seed: Annotated[
        list[str] | None,
        typer.Option("--seed", help="Seed URL to start browsing from. Repeatable."),
    ] = None,
) -> None:
    """Run a research task. Not yet implemented."""
    del question, seed
    typer.echo("scout.research is not implemented yet.", err=True)
    raise typer.Exit(code=2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
