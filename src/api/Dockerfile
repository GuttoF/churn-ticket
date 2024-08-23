FROM python:3.12.4-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libc-dev \
    libffi-dev \
    build-essential \
    && apt-get clean

# Install poetry
RUN pip install poetry


# Copy the current directory contents into the container at /app
COPY ../.. /app

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without doc,dev,jupyter-notebooks,ml_steps,eda

# PORT
EXPOSE 8000

CMD ["gunicorn", "src.api.main:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "6", "--bind", "0.0.0.0:8000"]