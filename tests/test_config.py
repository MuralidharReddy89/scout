from __future__ import annotations

import pytest
from pydantic import ValidationError

from scout.config import (
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_WALL_SECONDS,
    DEFAULT_MODEL,
    Config,
)


def test_defaults_when_no_env() -> None:
    c = Config.from_env(env={})
    assert c.anthropic_api_key is None
    assert c.llm_model == DEFAULT_MODEL
    assert c.budget_limits.max_tokens is None
    assert c.budget_limits.max_usd is None
    assert c.budget_limits.max_pages == DEFAULT_MAX_PAGES
    assert c.budget_limits.max_wall_seconds == DEFAULT_MAX_WALL_SECONDS


def test_reads_all_env_vars() -> None:
    c = Config.from_env(
        env={
            "ANTHROPIC_API_KEY": "sk-test-redacted",
            "SCOUT_LLM_MODEL": "claude-test",
            "SCOUT_MAX_TOKENS": "5000",
            "SCOUT_MAX_USD": "1.50",
            "SCOUT_MAX_PAGES": "10",
            "SCOUT_MAX_WALL_SECONDS": "60",
        }
    )
    assert c.anthropic_api_key == "sk-test-redacted"
    assert c.llm_model == "claude-test"
    assert c.budget_limits.max_tokens == 5000
    assert c.budget_limits.max_usd == 1.50
    assert c.budget_limits.max_pages == 10
    assert c.budget_limits.max_wall_seconds == 60.0


def test_empty_api_key_normalizes_to_none() -> None:
    c = Config.from_env(env={"ANTHROPIC_API_KEY": ""})
    assert c.anthropic_api_key is None


def test_whitespace_only_api_key_normalizes_to_none() -> None:
    c = Config.from_env(env={"ANTHROPIC_API_KEY": "  \n\t "})
    assert c.anthropic_api_key is None


def test_api_key_is_stripped_of_surrounding_whitespace() -> None:
    c = Config.from_env(env={"ANTHROPIC_API_KEY": "\n  sk-test-redacted  \n"})
    assert c.anthropic_api_key == "sk-test-redacted"


def test_empty_model_normalizes_to_default() -> None:
    c = Config.from_env(env={"SCOUT_LLM_MODEL": ""})
    assert c.llm_model == DEFAULT_MODEL


def test_invalid_int_env_var_raises() -> None:
    with pytest.raises(ValueError, match="SCOUT_MAX_TOKENS"):
        Config.from_env(env={"SCOUT_MAX_TOKENS": "not-a-number"})


def test_invalid_float_env_var_raises() -> None:
    with pytest.raises(ValueError, match="SCOUT_MAX_USD"):
        Config.from_env(env={"SCOUT_MAX_USD": "abc"})


def test_empty_numeric_env_var_falls_back_to_default() -> None:
    c = Config.from_env(env={"SCOUT_MAX_PAGES": ""})
    assert c.budget_limits.max_pages == DEFAULT_MAX_PAGES


def test_config_is_frozen() -> None:
    c = Config()
    with pytest.raises(ValidationError):
        c.llm_model = "mutated"


def test_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"llm_model": "x", "unexpected": 1})
