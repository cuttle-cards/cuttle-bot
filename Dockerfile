# syntax=docker/dockerfile:1

FROM node:20-slim AS web-build
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY game ./game
COPY rl ./rl
COPY server ./server

COPY --from=web-build /app/web/dist ./web/dist

EXPOSE 8000

CMD sh -c "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}"
