# Terminal 1: start evaluator (Ollama Cloud setup)
  cd ../evaluator
  EVALUATOR_BASE_URL=http://host.docker.internal:11434/v1 \
  EVALUATOR_MODEL=kimi-k2.5:cloud \
  docker compose up

  # Terminal 2: start level4
  cd level4
  docker compose up