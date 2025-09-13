# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000
CMD ["sh","-c","uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
