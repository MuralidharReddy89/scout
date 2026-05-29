from __future__ import annotations

import pytest
from pydantic import ValidationError

from scout.llm.base import (
    Completion,
    Message,
    TextBlock,
    TokenUsage,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)


def test_text_block_defaults_type_to_text() -> None:
    b = TextBlock(text="hello")
    assert b.type == "text"
    assert b.text == "hello"


def test_tool_use_block_defaults_input_to_empty_dict() -> None:
    b = ToolUseBlock(id="t1", name="fetch")
    assert b.type == "tool_use"
    assert b.input == {}


def test_tool_use_block_rejects_empty_identifiers() -> None:
    with pytest.raises(ValidationError):
        ToolUseBlock(id="", name="fetch")
    with pytest.raises(ValidationError):
        ToolUseBlock(id="t1", name="")


def test_tool_result_block_defaults_is_error_false() -> None:
    b = ToolResultBlock(tool_use_id="t1", content="ok")
    assert b.is_error is False
    assert b.type == "tool_result"


def test_message_requires_non_empty_content() -> None:
    with pytest.raises(ValidationError):
        Message(role="user", content=())


def test_message_content_uses_discriminator_when_parsed_from_dict() -> None:
    m = Message.model_validate(
        {
            "role": "assistant",
            "content": (
                {"type": "text", "text": "thinking..."},
                {"type": "tool_use", "id": "t1", "name": "fetch", "input": {"url": "x"}},
            ),
        }
    )
    assert isinstance(m.content[0], TextBlock)
    assert isinstance(m.content[1], ToolUseBlock)
    assert m.content[1].input == {"url": "x"}


def test_tool_definition_requires_name_and_description() -> None:
    td = ToolDefinition(name="fetch", description="get a url", input_schema={"type": "object"})
    assert td.name == "fetch"
    with pytest.raises(ValidationError):
        ToolDefinition(name="", description="d", input_schema={})


def test_token_usage_total_and_non_negative() -> None:
    u = TokenUsage(input_tokens=10, output_tokens=5)
    assert u.total_tokens == 15
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=-1, output_tokens=0)


def _completion(
    *,
    blocks: tuple[TextBlock | ToolUseBlock, ...],
    stop: str = "end_turn",
) -> Completion:
    return Completion(
        content=blocks,
        stop_reason=stop,  # type: ignore[arg-type]
        usage=TokenUsage(input_tokens=1, output_tokens=2),
        model="fake-model",
    )


def test_completion_text_concatenates_text_blocks_only() -> None:
    c = _completion(
        blocks=(
            TextBlock(text="hello "),
            ToolUseBlock(id="t1", name="fetch"),
            TextBlock(text="world"),
        ),
        stop="tool_use",
    )
    assert c.text == "hello world"


def test_completion_tool_uses_returns_only_tool_use_blocks() -> None:
    tu = ToolUseBlock(id="t1", name="fetch", input={"u": "x"})
    c = _completion(blocks=(TextBlock(text="x"), tu), stop="tool_use")
    assert c.tool_uses == (tu,)


def test_completion_rejects_tool_result_block_in_content() -> None:
    with pytest.raises(ValidationError):
        Completion(
            content=(ToolResultBlock(tool_use_id="t1", content="x"),),  # type: ignore[arg-type]
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=0, output_tokens=0),
            model="fake-model",
        )


def test_completion_rejects_unknown_stop_reason() -> None:
    with pytest.raises(ValidationError):
        Completion(
            content=(TextBlock(text="x"),),
            stop_reason="exploded",  # type: ignore[arg-type]
            usage=TokenUsage(input_tokens=0, output_tokens=0),
            model="fake-model",
        )


def test_models_are_frozen() -> None:
    b = TextBlock(text="hello")
    with pytest.raises(ValidationError):
        b.text = "mutated"
