#!/bin/bash
# =============================================================================
# Level 4 — Model Setup Script
#
# Checks for required Ollama models and downloads any that are missing.
# Run this BEFORE starting the Level 4 pipeline.
#
# Required models:
#   qwen3:8b     — Engagement Manager (fast decomposition + review)
#   qwen3:14b    — Market Researcher, Financial Analyst, Risk Analyst
#   qwen3.5:27b  — Strategy Consultant (256K context, superior writing)
#
# The Evaluator uses Claude Opus 4.6 via the Anthropic API (no local model).
# =============================================================================

set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

REQUIRED_MODELS=(
    "qwen3:8b"
    "qwen3:14b"
    "qwen3.5:27b"
)

echo "==========================================="
echo " Level 4 — Ollama Model Setup"
echo "==========================================="
echo ""

# Check if Ollama is running
echo "Checking Ollama at ${OLLAMA_HOST}..."
if ! curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running at ${OLLAMA_HOST}"
    echo "Start Ollama first:  ollama serve"
    exit 1
fi
echo "Ollama is running."
echo ""

# Get list of available models
AVAILABLE=$(curl -sf "${OLLAMA_HOST}/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(m['name'])
" 2>/dev/null || echo "")

MISSING=()
PRESENT=()

for model in "${REQUIRED_MODELS[@]}"; do
    if echo "$AVAILABLE" | grep -q "^${model}$"; then
        PRESENT+=("$model")
    else
        # Also check without exact match (e.g., qwen3:8b might show as qwen3:8b-q4_K_M)
        if echo "$AVAILABLE" | grep -q "^${model}"; then
            PRESENT+=("$model")
        else
            MISSING+=("$model")
        fi
    fi
done

echo "Status:"
for model in "${PRESENT[@]}"; do
    echo "  [OK] $model — already available"
done
for model in "${MISSING[@]}"; do
    echo "  [--] $model — needs download"
done
echo ""

if [ ${#MISSING[@]} -eq 0 ]; then
    echo "All required models are already available!"
else
    echo "Downloading ${#MISSING[@]} missing model(s)..."
    echo ""
    for model in "${MISSING[@]}"; do
        echo "Pulling $model..."
        ollama pull "$model"
        if [ $? -eq 0 ]; then
            echo "  [OK] $model downloaded successfully"
        else
            echo "  [FAIL] Failed to download $model"
            echo ""
            echo "If the download failed, try manually:  ollama pull $model"
            exit 1
        fi
        echo ""
    done
    echo "All models downloaded successfully!"
fi

echo ""
echo "==========================================="
echo " Evaluator Setup"
echo "==========================================="
echo ""
echo "The Evaluator uses Claude Opus 4.6 via the Anthropic API."
echo "It requires the ANTHROPIC_API_KEY environment variable."
echo ""

if [ -z "${ANTHROPIC_API_KEY}" ]; then
    echo "[WARNING] ANTHROPIC_API_KEY is not set."
    echo "The pipeline will run without the Evaluator."
    echo ""
    echo "To enable it, set the key:"
    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
    echo "Or add it to a .env file in the level4 directory."
else
    echo "[OK] ANTHROPIC_API_KEY is configured."
fi

echo ""
echo "==========================================="
echo " Ready to run Level 4"
echo "==========================================="
echo ""
echo "Start with Docker:"
echo "  cd level4 && docker compose up --build"
echo ""
echo "Or run directly:"
echo "  cd level4/backend && uvicorn main:app --host 0.0.0.0 --port 8000"
echo ""
