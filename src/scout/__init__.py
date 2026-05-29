from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scout")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from scout.cli import main
from scout.models import Citation, Finding, Report

__all__ = ["Citation", "Finding", "Report", "__version__", "main"]
