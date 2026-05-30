from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from anthropic.types import TextBlock as AnthropicTextBlock
from anthropic.types import ToolUseBlock as AnthropicToolUseBlock

from scout.llm.anthropic_client import AnthropicClient
from scout.llm.base import (
    LLMClient,
    Message,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)


def _text(text: str) -> AnthropicTextBlock:
    return AnthropicTextBlock(type="text", text=text)


def _tool_use(*, id: str, name: str, input: dict[str, Any]) -> AnthropicToolUseBlock:
    return AnthropicToolUseBlock(type="tool_use", id=id, name=name, input=input)


def _stub_client(response: Any) -> MagicMock:
    """Build a stand-in for ``anthropic.Anthropic`` with a recordable ``messages.create``."""
    client = MagicMock()
    client.messages.create.return_value = response
    return client


def _resp(
    *,
    content: list[Any],
    stop_reason: str = "end_turn",
    input_tokens: int = 3,
    output_tokens: int = 4,
    model: str = "claude-test-model",
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        model=model,
    )


def _user(text: str) -> Message:
    return Message(role="user", content=(TextBlock(text=text),))


def test_satisfies_llm_client_protocol() -> None:
    c = AnthropicClient(
        api_key="sk-test", model="claude-test", client=_stub_client(_resp(content=[]))
    )
    assert isinstance(c, LLMClient)


def test_rejects_empty_api_key_or_model() -> None:
    with pytest.raises(ValueError, match="api_key"):
        AnthropicClient(api_key="", model="m", client=cast(Any, MagicMock()))
    with pytest.raises(ValueError, match="model"):
        AnthropicClient(api_key="k", model="", client=cast(Any, MagicMock()))


def test_rejects_whitespace_only_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        AnthropicClient(api_key="   \n\t  ", model="m", client=cast(Any, MagicMock()))


def test_rejects_api_key_with_internal_whitespace() -> None:
    with pytest.raises(ValueError, match="internal whitespace"):
        AnthropicClient(api_key="sk-abc\ndef", model="m", client=cast(Any, MagicMock()))


def test_strips_surrounding_whitespace_when_constructing_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no explicit client is passed, the api_key handed to the SDK is stripped.

    Catches the PowerShell multi-line-assignment failure mode where the env var
    captured a literal leading newline and the Anthropic SDK then produced
    httpx ``LocalProtocolError: Illegal header value``.
    """
    captured: dict[str, Any] = {}

    class _FakeAnthropic:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    AnthropicClient(api_key="  \n sk-padded \n ", model="m")
    assert captured["api_key"] == "sk-padded"


def test_complete_validates_inputs() -> None:
    client = AnthropicClient(api_key="k", model="m", client=_stub_client(_resp(content=[])))
    with pytest.raises(ValueError, match="max_tokens"):
        client.complete([_user("hi")], max_tokens=0)
    with pytest.raises(ValueError, match="messages"):
        client.complete([], max_tokens=10)


def test_request_payload_serializes_all_block_types() -> None:
    stub = _stub_client(_resp(content=[_text("ok")]))
    client = AnthropicClient(api_key="k", model="m", client=stub)
    msgs = [
        Message(role="user", content=(TextBlock(text="hello"),)),
        Message(
            role="assistant",
            content=(ToolUseBlock(id="t1", name="fetch", input={"u": "x"}),),
        ),
        Message(
            role="user",
            content=(ToolResultBlock(tool_use_id="t1", content="body"),),
        ),
    ]
    tool = ToolDefinition(name="fetch", description="d", input_schema={"type": "object"})
    client.complete(msgs, max_tokens=128, system="you are scout", tools=[tool])

    kwargs = stub.messages.create.call_args.kwargs
    assert kwargs["model"] == "m"
    assert kwargs["max_tokens"] == 128
    assert kwargs["system"] == "you are scout"
    assert kwargs["tools"] == [
        {"name": "fetch", "description": "d", "input_schema": {"type": "object"}}
    ]
    assert kwargs["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "fetch", "input": {"u": "x"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "body",
                    "is_error": False,
                }
            ],
        },
    ]


def test_optional_system_and_tools_omitted_when_absent() -> None:
    stub = _stub_client(_resp(content=[_text("hi")]))
    client = AnthropicClient(api_key="k", model="m", client=stub)
    client.complete([_user("q")], max_tokens=10)
    kwargs = stub.messages.create.call_args.kwargs
    assert "system" not in kwargs
    assert "tools" not in kwargs


def test_response_parses_text_and_tool_use_blocks() -> None:
    resp = _resp(
        content=[
            _text("thinking..."),
            _tool_use(id="t1", name="fetch", input={"u": "x"}),
        ],
        stop_reason="tool_use",
        input_tokens=11,
        output_tokens=22,
    )
    client = AnthropicClient(api_key="k", model="m", client=_stub_client(resp))
    out = client.complete([_user("q")], max_tokens=10)
    assert out.stop_reason == "tool_use"
    assert out.usage.input_tokens == 11
    assert out.usage.output_tokens == 22
    assert out.text == "thinking..."
    assert out.tool_uses == (ToolUseBlock(id="t1", name="fetch", input={"u": "x"}),)


def test_response_rejects_unknown_block_type() -> None:
    resp = _resp(content=[SimpleNamespace(type="thinking", text="...")])
    client = AnthropicClient(api_key="k", model="m", client=_stub_client(resp))
    with pytest.raises(ValueError, match="unsupported anthropic content block"):
        client.complete([_user("q")], max_tokens=10)


def test_response_rejects_unknown_stop_reason() -> None:
    resp = _resp(content=[_text("x")], stop_reason="refusal")
    client = AnthropicClient(api_key="k", model="m", client=_stub_client(resp))
    with pytest.raises(ValueError, match="unsupported anthropic stop_reason"):
        client.complete([_user("q")], max_tokens=10)


def test_response_rejects_null_stop_reason() -> None:
    resp = _resp(content=[_text("x")], stop_reason=cast(Any, None))
    client = AnthropicClient(api_key="k", model="m", client=_stub_client(resp))
    with pytest.raises(ValueError, match="unsupported anthropic stop_reason"):
        client.complete([_user("q")], max_tokens=10)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live integration test",
)
def test_live_round_trip_against_real_api() -> None:
    model = os.environ.get("SCOUT_INTEGRATION_MODEL", "claude-haiku-4-5-20251001")
    client = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"], model=model)
    out = client.complete(
        [_user("Reply with the single word: pong")],
        max_tokens=16,
    )
    assert out.text.strip() != ""
    assert out.usage.input_tokens > 0
    assert out.usage.output_tokens > 0
