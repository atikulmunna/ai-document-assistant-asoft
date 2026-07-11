FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, static UI, and the pre-built index (baked in at build time
# so the container never re-embeds the corpus at startup). The source PDFs are
# not copied: they are only needed to build the index, never to serve it.
COPY app ./app
COPY static ./static
COPY data ./data

# The container listens on PORT (default 8080); the host can override it.
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
