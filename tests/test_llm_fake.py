from __future__ import annotations

import pytest

from scout.llm.base import (
    Completion,
    LLMClient,
    Message,
    TextBlock,
    TokenUsage,
    ToolDefinition,
)
from scout.llm.fake import FakeLLMClient, FakeLLMExhausted


def _completion(text: str = "ok") -> Completion:
    return Completion(
        content=(TextBlock(text=text),),
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        model="fake-model",
    )


def _user_msg(text: str) -> Message:
    return Message(role="user", content=(TextBlock(text=text),))


def test_fake_satisfies_llm_client_protocol() -> None:
    assert isinstance(FakeLLMClient(), LLMClient)


def test_returns_scripted_responses_in_fifo_order() -> None:
    a, b = _completion("first"), _completion("second")
    fake = FakeLLMClient([a, b])
    assert fake.complete([_user_msg("q1")], max_tokens=10) is a
    assert fake.complete([_user_msg("q2")], max_tokens=10) is b


def test_records_call_inputs() -> None:
    fake = FakeLLMClient([_completion()])
    tool = ToolDefinition(name="fetch", description="d", input_schema={"type": "object"})
    msgs = [_user_msg("hello")]
    fake.complete(msgs, max_tokens=42, system="you are scout", tools=[tool])
    assert len(fake.calls) == 1
    rec = fake.calls[0]
    assert rec.messages == tuple(msgs)
    assert rec.system == "you are scout"
    assert rec.tools == (tool,)
    assert rec.max_tokens == 42


def test_records_call_even_when_script_exhausted() -> None:
    fake = FakeLLMClient()
    with pytest.raises(FakeLLMExhausted) as exc:
        fake.complete([_user_msg("q")], max_tokens=10)
    assert exc.value.call_index == 0
    assert len(fake.calls) == 1


def test_exhausted_exception_carries_correct_call_index() -> None:
    fake = FakeLLMClient([_completion()])
    fake.complete([_user_msg("q1")], max_tokens=10)
    with pytest.raises(FakeLLMExhausted) as exc:
        fake.complete([_user_msg("q2")], max_tokens=10)
    assert exc.value.call_index == 1


def test_remaining_decrements_with_each_call() -> None:
    fake = FakeLLMClient([_completion(), _completion()])
    assert fake.remaining == 2
    fake.complete([_user_msg("q")], max_tokens=10)
    assert fake.remaining == 1


def test_queue_appends_response_after_construction() -> None:
    fake = FakeLLMClient()
    extra = _completion("late")
    fake.queue(extra)
    assert fake.remaining == 1
    assert fake.complete([_user_msg("q")], max_tokens=10) is extra


def test_default_construction_uses_empty_script() -> None:
    fake = FakeLLMClient()
    assert fake.remaining == 0
    assert fake.calls == []
