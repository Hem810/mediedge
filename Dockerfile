FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . .

# Create persistent data volume mount points
RUN mkdir -p /app/uploads
VOLUME ["/app/uploads", "/app/data"]

# IMPORTANT: This container talks to a separate Ollama server.
# In docker-compose.yml, OLLAMA_HOST is set to http://ollama:11434
# pointing to the bundled ollama service.
ENV OLLAMA_HOST=http://ollama:11434

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
