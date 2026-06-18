#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON:-python}"
LEVEL_ARGS=(--level level_1 --level level_2 --level level_3)

# Optional extra args, for example:
#   RUN_QA_EXTRA_ARGS="--overwrite-results --overwrite-evaluations" ./run_all_qa_experiments.sh
if [[ -n "${RUN_QA_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${RUN_QA_EXTRA_ARGS})
else
  EXTRA_ARGS=()
fi

run_experiment() {
  local label="$1"
  shift

  local cmd=(
    "${PYTHON_BIN}"
    -B
    run_qa_experiment.py
    "${LEVEL_ARGS[@]}"
    "$@"
    "${EXTRA_ARGS[@]}"
  )

  echo "==> Running ${label} on level_1, level_2, level_3"
  "${cmd[@]}"
}

run_experiment   "Llama 3.2 1B"   --generator llama   --model meta-llama/Llama-3.2-1B-Instruct

run_experiment   "GPT 5.4 mini"   --generator openai   --model gpt-5.4-mini
