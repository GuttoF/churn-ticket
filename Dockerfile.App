FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

RUN useradd -m -u 1000 appuser

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY src /app/src
COPY main.py pyproject.toml uv.lock /app/

RUN mkdir -p /app/data/processed
COPY data/processed/ /app/data/processed/


RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

RUN chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT []

EXPOSE 8501

CMD ["streamlit", "run", "src/app/main.py"]
