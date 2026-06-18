#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" -B extract_sample_dataset.py \
  --input-dir dataset_counter \
  --output-dir dataset_counter_sample
