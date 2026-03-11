#!/bin/sh
set -e

OLLAMA_BASE="${OLLAMA_HOST:-http://ollama:11434}"

echo "[level2] Waiting for Ollama at ${OLLAMA_BASE}..."
until curl -sf "${OLLAMA_BASE}/api/tags" > /dev/null 2>&1; do
    sleep 2
done
echo "[level2] Ollama is ready."

# List the models this level expects
echo "[level2] Expected models:"
echo "  Engagement Manager : ${ENGAGEMENT_MANAGER_MODEL:-qwen3:8b}"
echo "  Market Researcher  : ${MARKET_RESEARCHER_MODEL:-qwen3:14b}"
echo "  Strategy Consultant: ${STRATEGY_CONSULTANT_MODEL:-qwen3:32b}"
echo "  Evaluator          : ${EVALUATOR_MODEL:-qwen3:14b}"

echo "[level2] Available models on Ollama:"
curl -sf "${OLLAMA_BASE}/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(f\"  - {m['name']}\")
" 2>/dev/null || echo "  (could not list models)"

echo "[level2] Starting backend on port 8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
