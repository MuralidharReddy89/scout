from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from scout.browser.base import BrowserError, BrowserTool, Page
from scout.budget import Budget, BudgetExceeded
from scout.llm.base import (
    LLMClient,
    Message,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from scout.models import Citation, Finding, Report

DEFAULT_SYSTEM_PROMPT = (
    "You are scout, a careful research agent. Use fetch_url to read web pages "
    "and record_finding to capture each grounded claim with a verbatim quote "
    "from the source. Do not record findings you cannot quote. When the user's "
    "question is answered, stop calling tools and reply with a short summary."
)

FETCH_URL_TOOL = ToolDefinition(
    name="fetch_url",
    description="Fetch a web page by URL and return a text preview.",
    input_schema={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Absolute http(s) URL."}},
        "required": ["url"],
    },
)

RECORD_FINDING_TOOL = ToolDefinition(
    name="record_finding",
    description=(
        "Record one grounded finding. 'source_url' must be a URL previously "
        "returned by fetch_url; 'quote' must appear verbatim on that page."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "source_url": {"type": "string"},
            "quote": {"type": "string"},
        },
        "required": ["claim", "source_url", "quote"],
    },
)


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class Researcher:
    """Drives an LLM through tool-use turns to answer a research question.

    Synchronous loop: each iteration calls :meth:`LLMClient.complete`, executes
    any requested tools against the supplied :class:`BrowserTool`, and feeds
    results back as ``ToolResultBlock``s. Terminates on the first of: clean
    end_turn from the model, ``max_iterations`` reached, or a
    :class:`BudgetExceeded` raised by the supplied :class:`Budget`.
    """

    def __init__(
        self,
        llm: LLMClient,
        browser: BrowserTool,
        budget: Budget,
        *,
        max_iterations: int = 12,
        max_output_tokens: int = 1024,
        text_preview_chars: int = 2000,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if text_preview_chars < 0:
            raise ValueError("text_preview_chars must be non-negative")
        self._llm = llm
        self._browser = browser
        self._budget = budget
        self._max_iterations = max_iterations
        self._max_output_tokens = max_output_tokens
        self._text_preview_chars = text_preview_chars
        self._system_prompt = system_prompt
        self._clock = clock
        self._fetched: dict[str, Page] = {}

    def research(self, question: str) -> Report:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        started_at = self._clock()
        messages: list[Message] = [Message(role="user", content=(TextBlock(text=question),))]
        findings: list[Finding] = []
        tools = (FETCH_URL_TOOL, RECORD_FINDING_TOOL)
        summary = ""
        model_name = ""
        truncated = False
        over_budget = False

        for _ in range(self._max_iterations):
            try:
                self._budget.check_wall_clock()
            except BudgetExceeded:
                over_budget = True
                break
            completion = self._llm.complete(
                messages,
                max_tokens=self._max_output_tokens,
                system=self._system_prompt,
                tools=tools,
            )
            model_name = completion.model
            try:
                self._budget.consume_tokens(completion.usage.total_tokens)
            except BudgetExceeded:
                over_budget = True
                break
            messages.append(Message(role="assistant", content=completion.content))
            if not completion.tool_uses:
                summary = completion.text
                break
            results, budget_hit = self._run_tools(completion.tool_uses, findings)
            messages.append(Message(role="user", content=tuple(results)))
            if budget_hit:
                over_budget = True
                break
        else:
            truncated = True

        return Report(
            question=question,
            findings=tuple(findings),
            model=model_name or "unknown",
            started_at=started_at,
            finished_at=self._clock(),
            summary=summary,
            truncated=truncated,
            over_budget=over_budget,
        )

    def _run_tools(
        self, calls: tuple[ToolUseBlock, ...], findings: list[Finding]
    ) -> tuple[list[ToolResultBlock], bool]:
        """Execute each tool call; return the result blocks and a budget-hit flag."""
        results: list[ToolResultBlock] = []
        budget_hit = False
        for call in calls:
            if call.name == FETCH_URL_TOOL.name:
                content, is_error, hit = self._do_fetch(call.input)
            elif call.name == RECORD_FINDING_TOOL.name:
                content, is_error = self._do_record(call.input, findings)
                hit = False
            else:
                content, is_error, hit = f"unknown tool: {call.name}", True, False
            budget_hit = budget_hit or hit
            results.append(ToolResultBlock(tool_use_id=call.id, content=content, is_error=is_error))
        return results, budget_hit

    def _do_fetch(self, args: dict[str, Any]) -> tuple[str, bool, bool]:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return "fetch_url requires a non-empty 'url' string", True, False
        try:
            self._budget.consume_page()
        except BudgetExceeded:
            return "page budget exhausted", True, True
        try:
            page = self._browser.fetch(url)
        except BrowserError as exc:
            return f"fetch failed for {url}: {exc}", True, False
        self._fetched[page.url] = page
        self._fetched[page.final_url] = page
        preview = page.text[: self._text_preview_chars]
        return (
            (
                f"final_url={page.final_url}\n"
                f"status_code={page.status_code}\n"
                f"title={page.title}\n"
                f"text_length={len(page.text)}\n\n"
                f"{preview}"
            ),
            False,
            False,
        )

    def _do_record(self, args: dict[str, Any], findings: list[Finding]) -> tuple[str, bool]:
        claim = args.get("claim")
        source_url = args.get("source_url")
        quote = args.get("quote")
        for name, value in (("claim", claim), ("source_url", source_url), ("quote", quote)):
            if not isinstance(value, str) or not value.strip():
                return f"record_finding requires a non-empty '{name}' string", True
        assert isinstance(claim, str) and isinstance(source_url, str) and isinstance(quote, str)
        if source_url not in self._fetched:
            return (f"source_url {source_url!r} has not been fetched; call fetch_url first"), True
        page = self._fetched[source_url]
        if quote not in page.text:
            return (
                f"quote not found verbatim on {source_url}; "
                "copy an exact substring from the page text"
            ), True
        try:
            citation = Citation.model_validate(
                {"url": source_url, "quote": quote, "retrieved_at": self._clock()}
            )
            finding = Finding(claim=claim, citations=(citation,))
        except ValidationError as exc:
            return f"validation failed: {exc.errors(include_url=False)}", True
        findings.append(finding)
        return f"ok, finding_count={len(findings)}", False
