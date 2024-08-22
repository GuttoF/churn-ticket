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
COPY . /app

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi --without doc,dev,jupyter-notebooks,ml_steps,eda

# FASTAPI PORT
EXPOSE 8000

# Streamlit PORT
EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "src/app/main.py"]
