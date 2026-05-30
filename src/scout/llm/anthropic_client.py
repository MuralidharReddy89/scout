from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast, get_args

from anthropic.types import TextBlock as AnthropicTextBlock
from anthropic.types import ToolUseBlock as AnthropicToolUseBlock

from scout.llm.base import (
    Completion,
    Message,
    StopReason,
    TextBlock,
    TokenUsage,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import Message as AnthropicMessage

_ALLOWED_STOP_REASONS: frozenset[str] = frozenset(get_args(StopReason))


class AnthropicClient:
    """``LLMClient`` implementation backed by the official ``anthropic`` SDK.

    Conversions are deliberately narrow: only the content-block shapes the
    rest of the codebase models (``text``, ``tool_use``, ``tool_result``) are
    accepted on input or produced on output. Server-side tool blocks,
    thinking blocks, and other SDK-internal variants raise so that callers
    notice rather than silently dropping data.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        client: Anthropic | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        stripped_key = api_key.strip()
        if not stripped_key:
            raise ValueError("api_key must be a non-empty string")
        if any(ch.isspace() for ch in stripped_key):
            raise ValueError(
                "api_key contains internal whitespace; check for embedded "
                "newlines or spaces (often caused by multi-line shell "
                "assignments such as a PowerShell here-string)"
            )
        if not model:
            raise ValueError("model must be a non-empty string")
        self._model = model
        if client is None:
            from anthropic import Anthropic as _Anthropic

            client = _Anthropic(api_key=stripped_key)
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        system: str | None = None,
        tools: Sequence[ToolDefinition] = (),
    ) -> Completion:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not messages:
            raise ValueError("messages must not be empty")

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [_message_to_param(m) for m in messages],
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_tool_to_param(t) for t in tools]

        response = self._client.messages.create(**kwargs)
        return _response_to_completion(cast("AnthropicMessage", response))


def _message_to_param(msg: Message) -> dict[str, Any]:
    return {
        "role": msg.role,
        "content": [_block_to_param(b) for b in msg.content],
    }


def _block_to_param(block: TextBlock | ToolUseBlock | ToolResultBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    return {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": block.content,
        "is_error": block.is_error,
    }


def _tool_to_param(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _response_to_completion(resp: AnthropicMessage) -> Completion:
    blocks: list[TextBlock | ToolUseBlock] = []
    for block in resp.content:
        if isinstance(block, AnthropicTextBlock):
            blocks.append(TextBlock(text=block.text))
        elif isinstance(block, AnthropicToolUseBlock):
            blocks.append(
                ToolUseBlock(
                    id=block.id,
                    name=block.name,
                    input=dict(block.input),
                )
            )
        else:
            raise ValueError(
                f"unsupported anthropic content block type: {getattr(block, 'type', None)!r}"
            )

    stop = resp.stop_reason
    if stop is None or stop not in _ALLOWED_STOP_REASONS:
        raise ValueError(f"unsupported anthropic stop_reason: {stop!r}")

    return Completion(
        content=tuple(blocks),
        stop_reason=cast(StopReason, stop),
        usage=TokenUsage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        ),
        model=str(resp.model),
    )
