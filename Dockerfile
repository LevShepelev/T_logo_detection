# Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml /app/
RUN pip3 install --upgrade pip && pip3 install -e .

# Копируем код сервиса и веса
COPY app/ /app/app/
COPY models/ /app/models/

# (опц.) если используете ultralytics — установите здесь:
RUN pip3 install "ultralytics>=8.3.0"

EXPOSE 8000
CMD ["python3", "-m", "app.main"]
