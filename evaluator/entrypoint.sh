#!/bin/sh
set -e

MODEL="${EVALUATOR_MODEL:-gemini-2.5-flash}"
BASE_URL="${EVALUATOR_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"

if [ -z "${EVALUATOR_API_KEY}" ]; then
    echo "[evaluator] WARNING: EVALUATOR_API_KEY is not set."
    echo "[evaluator] /evaluate will return 503 until a key is provided."
fi

echo "[evaluator] Model:    ${MODEL}"
echo "[evaluator] Base URL: ${BASE_URL}"
echo "[evaluator] Starting backend on port 8000..."

exec uvicorn main:app --host 0.0.0.0 --port 8000
