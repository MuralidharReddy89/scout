from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scout")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from scout.agent import Researcher
from scout.budget import Budget, BudgetExceeded, BudgetLimits, BudgetUsage
from scout.cli import main
from scout.config import Config
from scout.models import Citation, Finding, Report

__all__ = [
    "Budget",
    "BudgetExceeded",
    "BudgetLimits",
    "BudgetUsage",
    "Citation",
    "Config",
    "Finding",
    "Report",
    "Researcher",
    "__version__",
    "main",
]
