from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["user", "assistant"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]


class _Frozen(BaseModel):
    """Common base for LLM payload models: immutable, strict, no extras."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )


class TextBlock(_Frozen):
    """A plain-text content block emitted by or sent to the model."""

    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(_Frozen):
    """A request from the assistant to invoke a tool.

    ``id`` is the model-supplied correlation identifier; the eventual
    ``ToolResultBlock`` must echo it back in ``tool_use_id``.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(_Frozen):
    """The user-side result of a previously-requested tool call."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = Field(min_length=1)
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class Message(_Frozen):
    """One turn in the conversation transcript.

    The ``system`` prompt is passed separately to :meth:`LLMClient.complete`
    rather than as a role here, matching the Anthropic message format.
    """

    role: Role
    content: tuple[ContentBlock, ...] = Field(min_length=1)


class ToolDefinition(_Frozen):
    """Schema description of a tool the assistant may invoke."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]


class TokenUsage(_Frozen):
    """Token accounting for a single completion."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Completion(_Frozen):
    """The model's response to one :meth:`LLMClient.complete` call.

    Assistants never emit ``ToolResultBlock`` — those are user-side — so
    ``content`` is restricted to text and tool-use blocks.
    """

    content: tuple[
        Annotated[TextBlock | ToolUseBlock, Field(discriminator="type")],
        ...,
    ]
    stop_reason: StopReason
    usage: TokenUsage
    model: str = Field(min_length=1)

    @property
    def text(self) -> str:
        """Concatenate all text blocks; convenience for tests and logs."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> tuple[ToolUseBlock, ...]:
        """All tool-use blocks in order."""
        return tuple(b for b in self.content if isinstance(b, ToolUseBlock))


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic chat completion interface.

    Implementations are responsible for translating to and from their
    underlying SDK. The agent loop interacts only through this protocol so
    that ``FakeLLMClient`` can stand in during tests.
    """

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        system: str | None = None,
        tools: Sequence[ToolDefinition] = (),
    ) -> Completion: ...
