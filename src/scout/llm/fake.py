from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from scout.llm.base import Completion, Message, ToolDefinition


class FakeLLMExhausted(RuntimeError):
    """Raised when a ``FakeLLMClient`` is called past its scripted responses."""

    def __init__(self, call_index: int) -> None:
        super().__init__(
            f"FakeLLMClient script exhausted: call #{call_index} has no scripted response"
        )
        self.call_index = call_index


@dataclass(frozen=True)
class RecordedCall:
    """Snapshot of one ``complete`` invocation, for assertions in tests."""

    messages: tuple[Message, ...]
    system: str | None
    tools: tuple[ToolDefinition, ...]
    max_tokens: int


class FakeLLMClient:
    """Scripted, deterministic stand-in for the ``LLMClient`` protocol.

    Construct with an iterable of ``Completion`` objects. Each call to
    :meth:`complete` pops the next scripted response (FIFO) and records
    the inputs for later inspection via :attr:`calls`. When the script is
    exhausted, the next call raises :class:`FakeLLMExhausted`.

    Used in agent-loop tests to assert that the orchestrator constructs
    the expected prompts and reacts correctly to fixed model output.
    """

    def __init__(self, responses: Iterable[Completion] = ()) -> None:
        self._responses: list[Completion] = list(responses)
        self.calls: list[RecordedCall] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        system: str | None = None,
        tools: Sequence[ToolDefinition] = (),
    ) -> Completion:
        idx = len(self.calls)
        self.calls.append(
            RecordedCall(
                messages=tuple(messages),
                system=system,
                tools=tuple(tools),
                max_tokens=max_tokens,
            )
        )
        if not self._responses:
            raise FakeLLMExhausted(idx)
        return self._responses.pop(0)

    @property
    def remaining(self) -> int:
        """How many scripted responses have not yet been consumed."""
        return len(self._responses)

    def queue(self, response: Completion) -> None:
        """Append a response to the end of the script."""
        self._responses.append(response)
