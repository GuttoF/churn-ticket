FROM python:3.12.4-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libc-dev \
    libffi-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./
COPY src ./src

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without doc,dev,jupyter-notebooks,ml_steps,eda

RUN chown -R appuser:appuser /app

USER appuser

# PORT
EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "src/app/main.py"]
