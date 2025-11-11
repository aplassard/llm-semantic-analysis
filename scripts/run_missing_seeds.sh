#!/usr/bin/env bash
set -euo pipefail

PROMPTS="prompts/seed_prompt_corpus.jsonl"
OUTPUT_DIR="outputs"
CONFIGS=(
  "configs/openai/gpt-5-nano-high.yaml"
  "configs/openai/gpt-5-nano-low.yaml"
  "configs/openai/gpt-5-nano-medium.yaml"
  "configs/openai/gpt-5-nano-minimal.yaml"
  "configs/openai/gpt-5-high.yaml"
  "configs/openai/gpt-5-low.yaml"
  "configs/openai/gpt-5-medium.yaml"
  "configs/openai/gpt-5-minimal.yaml"
  "configs/anthropic/claude-4.5-sonnet-none.yaml"
  "configs/anthropic/claude-4.5-sonnet-low.yaml"
  "configs/anthropic/claude-4.5-sonnet-medium.yaml"
  "configs/anthropic/claude-4.5-sonnet-high.yaml"
  "configs/openrouter/gemini-2.5-pro-medium.yaml"
  "configs/openrouter/gemini-2.5-flash-lite-none.yaml"
  "configs/openrouter/gemini-2.5-flash-lite-low.yaml"
  "configs/openrouter/gemini-2.5-flash-lite-medium.yaml"
  "configs/openrouter/gemini-2.5-flash-lite-high.yaml"
)

for config in "${CONFIGS[@]}"; do
  base=$(basename "${config}" .yaml)
  output="${OUTPUT_DIR}/seed-${base}.jsonl"
  echo "=== Running ${base} ==="
  if [[ -f "${output}" ]]; then
    echo "Skipping ${base}: ${output} already exists"
    continue
  fi
  uv run --env-file .env python scripts/run_prompts.py \
    --prompts "${PROMPTS}" \
    --output "${output}" \
    --config "${config}" \
    --threads 12
  echo
  echo
  sleep 2
done
