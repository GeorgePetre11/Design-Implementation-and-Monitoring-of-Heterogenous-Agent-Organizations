#!/bin/sh
set -e

MODEL="${EVALUATOR_MODEL:-kimi-k2.5}"
BASE_URL="${EVALUATOR_BASE_URL:-https://api.moonshot.ai/v1}"

if [ -z "${EVALUATOR_API_KEY}" ]; then
    echo "[evaluator] WARNING: EVALUATOR_API_KEY is not set."
    echo "[evaluator] /evaluate will return 503 until a key is provided."
fi

echo "[evaluator] Model:    ${MODEL}"
echo "[evaluator] Base URL: ${BASE_URL}"
echo "[evaluator] Starting backend on port 8000..."

exec uvicorn main:app --host 0.0.0.0 --port 8000
