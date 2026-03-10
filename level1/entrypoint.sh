#!/bin/sh
set -e

OLLAMA_BASE="${OLLAMA_HOST:-http://ollama:11434}"
MODEL="${OLLAMA_MODEL:-qwen2.5:14b}"

echo "[level1] Waiting for Ollama at ${OLLAMA_BASE}..."
until curl -sf "${OLLAMA_BASE}/api/tags" > /dev/null 2>&1; do
    sleep 2
done
echo "[level1] Ollama is ready."
echo "[level1] Using model ${MODEL} from host Ollama."

echo "[level1] Starting backend on port 8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
