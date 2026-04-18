#!/bin/sh
set -e

OLLAMA_BASE="${OLLAMA_HOST:-http://ollama:11434}"

echo "[level4] Waiting for Ollama at ${OLLAMA_BASE}..."
until curl -sf "${OLLAMA_BASE}/api/tags" > /dev/null 2>&1; do
    sleep 2
done
echo "[level4] Ollama is ready."

# List the models this level expects
echo "[level4] Expected models:"
echo "  Engagement Manager : ${ENGAGEMENT_MANAGER_MODEL:-qwen3:8b}"
echo "  Market Researcher  : ${MARKET_RESEARCHER_MODEL:-qwen3:14b}"
echo "  Financial Analyst  : ${FINANCIAL_ANALYST_MODEL:-gpt-oss:20b}"
echo "  Risk Analyst       : ${RISK_ANALYST_MODEL:-qwen3:14b}"
echo "  Strategy Consultant: ${STRATEGY_CONSULTANT_MODEL:-qwen3.5:27b}"

echo "[level4] Available models on Ollama:"
curl -sf "${OLLAMA_BASE}/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(f\"  - {m['name']}\")
" 2>/dev/null || echo "  (could not list models)"

echo "[level4] Starting backend on port 8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
