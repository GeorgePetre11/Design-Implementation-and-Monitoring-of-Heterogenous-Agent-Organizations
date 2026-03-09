#!/bin/sh
set -e

OLLAMA_BASE="${OLLAMA_HOST:-http://ollama:11434}"
MODEL="${OLLAMA_MODEL:-qwen2.5:14b}"

echo "[level1] Waiting for Ollama at ${OLLAMA_BASE}..."
until curl -sf "${OLLAMA_BASE}/api/tags" > /dev/null 2>&1; do
    sleep 2
done
echo "[level1] Ollama is ready."

# Pull model only if not already present
if curl -sf "${OLLAMA_BASE}/api/tags" | grep -q "\"${MODEL}\""; then
    echo "[level1] Model ${MODEL} already available, skipping pull."
else
    echo "[level1] Pulling ${MODEL} — this may take several minutes on first run..."
    curl -s -X POST "${OLLAMA_BASE}/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"${MODEL}\"}" \
        --no-buffer | grep --line-buffered -o '"status":"[^"]*"' \
        | sed 's/"status":"//;s/"//' \
        | uniq
    echo "[level1] Model pull complete."
fi

echo "[level1] Starting backend on port 8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
