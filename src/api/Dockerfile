FROM python:3.12.4-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libc-dev \
    libffi-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./
COPY src/api ./src/api
COPY src/models ./src/models
COPY src/scalers ./src/scalers
COPY src/__init__.py ./src/__init__.py
COPY src/utils_feature_engineering.py ./src/utils_feature_engineering.py

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without doc,dev,jupyter-notebooks,ml_steps,eda

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["poetry", "run", "gunicorn", "src.api.main:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "4", "--bind", "0.0.0.0:8000", "--timeout", "120"]

