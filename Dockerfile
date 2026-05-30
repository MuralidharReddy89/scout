# syntax=docker/dockerfile:1.7

# ---- uv binary source --------------------------------------------------------
# Distroless image whose sole content is the `uv` and `uvx` static binaries.
# Pinned by digest so the build is byte-for-byte reproducible.
FROM ghcr.io/astral-sh/uv:0.9@sha256:538e0b39736e7feae937a65983e49d2ab75e1559d35041f9878b7b7e51de91e4 AS uv

# ---- runtime image -----------------------------------------------------------
# Official Playwright Python image: Ubuntu 24.04 (Noble) with Python 3.12,
# Chromium / Firefox / WebKit, and all browser system dependencies preinstalled.
# Pinned by digest of the multi-arch manifest list for v1.60.0-noble, matching
# the `playwright>=1.60.0` pin in pyproject.toml.
FROM mcr.microsoft.com/playwright/python:v1.60.0-noble@sha256:8ff591d613b01c884cc488339ed4318b4513eaf0c57a164a878ba49e70e3f384

# Copy the uv binaries from the uv stage; nothing else from that image is kept.
COPY --from=uv /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# Layer 1: install third-party dependencies only. Cached as long as
# pyproject.toml and uv.lock are unchanged.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Layer 2: install the project itself. Invalidated only when src/ changes.
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENTRYPOINT ["scout"]
CMD ["--help"]
