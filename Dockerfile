# Dockerfile (Poetry + lockfile only)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/root/.local/bin:${PATH}" 

# System deps + Poetry via pipx
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip pipx git ca-certificates libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && pipx install poetry

WORKDIR /app

# Copy manifests first for layer caching
COPY pyproject.toml poetry.lock ./

# Install exactly what's in poetry.lock
# (use build-arg to add groups like "--with runtime" if needed)
ARG POETRY_INSTALL_ARGS="--no-root --sync"
RUN poetry install $POETRY_INSTALL_ARGS

# Copy runtime code and weights
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000
CMD ["poetry", "run", "python", "-m", "app.main"]
