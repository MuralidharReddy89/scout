# scout

An LLM-driven web research agent built on Playwright and Python.

> **Status:** scaffolding — `v0.1.0` is in progress. The first release will ship a CLI
> (`scout research "question" --seed <url>`) plus a Python library
> (`from scout import Scout`) that drives a Playwright browser via an LLM-controlled
> tool-use loop and returns a structured JSON report with inline citations.

## Planned shape

- **Product surface:** CLI (`typer`) + library (`scout.Scout`)
- **Browser:** Playwright (chromium)
- **LLM:** pluggable — Anthropic adapter (default), OpenAI adapter, `FakeLLMClient` for CI
- **Output:** Pydantic `Report` with `Finding(claim, citations=[Citation(url, quote)])`
- **Budgets:** hard caps on tokens, USD, page count, wall-clock
- **Container:** SHA-pinned multi-stage Dockerfile on the official Playwright base image

## Status of this repository

This repo was renamed from a previous project on 2026-05-29 and rewritten from an
orphan `main` commit. The git history before that point is intentionally not
preserved.
