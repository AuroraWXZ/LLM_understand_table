#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON:-python3}"
QUESTIONS_ROOT="${COUNTER_QUESTIONS_ROOT:-question_counter}"
DATASET_DIR="${COUNTER_DATASET_DIR:-dataset_counter}"
RESULTS_ROOT="${COUNTER_RESULTS_ROOT:-counter_results}"
EVALUATION_ROOT="${COUNTER_EVALUATION_ROOT:-counter_evaluation}"
GPT_MODEL="${COUNTER_GPT_MODEL:-gpt-5.4-mini}"
LLAMA_MODEL="${COUNTER_LLAMA_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
EVALUATOR_MODEL="${COUNTER_EVALUATOR_MODEL:-gpt-4o-mini}"

if [[ -n "${COUNTER_QA_LEVELS:-}" ]]; then
  # shellcheck disable=SC2206
  LEVELS=(${COUNTER_QA_LEVELS})
else
  LEVELS=()
  for level_path in "${QUESTIONS_ROOT}"/level_*; do
    [[ -e "${level_path}" ]] || continue
    LEVELS+=("$(basename "${level_path}")")
  done
  if [[ ${#LEVELS[@]} -eq 0 ]]; then
    LEVELS=(level_1)
  fi
fi

LEVEL_ARGS=()
for level in "${LEVELS[@]}"; do
  LEVEL_ARGS+=(--level "${level}")
done

# Optional extra args, for example:
#   RUN_COUNTER_QA_EXTRA_ARGS="--overwrite-results --overwrite-evaluations" ./run_counter_qa_experiments.sh
if [[ -n "${RUN_COUNTER_QA_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${RUN_COUNTER_QA_EXTRA_ARGS})
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
    --questions-root "${QUESTIONS_ROOT}"
    --dataset-dir "${DATASET_DIR}"
    --results-root "${RESULTS_ROOT}"
    --evaluation-root "${EVALUATION_ROOT}"
    --evaluator-model "${EVALUATOR_MODEL}"
    "$@"
    "${EXTRA_ARGS[@]}"
  )

  echo "==> Running ${label} on ${LEVELS[*]}"
  "${cmd[@]}"
}

run_experiment   "Llama 3.2 1B"   --generator llama   --model "${LLAMA_MODEL}"

run_experiment   "GPT"   --generator openai   --model "${GPT_MODEL}"
