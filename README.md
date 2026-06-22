# LLM Understand Table

This repository builds a table-QA benchmark for testing whether language models answer from table evidence, and whether their answers stay grounded when the tables are changed into counterfactual versions.

The repo is set up to keep code, question templates, and small sample datasets in Git. Full raw data, generated questions, full datasets, model outputs, evaluations, reports, API keys, and chain-of-thought outputs are ignored.

## Where The Questions Are

Question assets live in two layers:

| Path | Purpose | Tracked in Git |
| --- | --- | --- |
| `question_template/level_1.csv` | Level 1 template questions | Yes |
| `question_template/level_2.csv` | Level 2 template questions | Yes |
| `question_template/level_3.csv` | Level 3 template questions | Yes |
| `question_template/seed/level_*.txt` | Seed values used by the generators | Yes |
| `question_template/question_generation/level_*.py` | Level-specific question generation logic | Yes |
| `questions/level_1`, `questions/level_2`, `questions/level_3` | Generated original-data questions | No, generated locally |
| `question_counter/level_1`, `question_counter/level_2`, `question_counter/level_3` | Generated counterfactual-data questions | No, generated locally |
| `raw_questions/level_*` | Optional raw text source for template extraction | No, ignored |

The generated question files are JSON arrays. Each record includes the question text, answer, explanation, level, template ID, and ground-truth table row references used during prompting and evaluation.

## Where The Sample Datasets Are

The small committed sample datasets are here:

| Path | Purpose |
| --- | --- |
| `dataset_sample/` | Original sample tables extracted from `dataset_clean/` |
| `dataset_counter_sample/` | Counterfactual sample tables extracted from `dataset_counter/` |

Both sample folders contain the same table set:

- `players.csv`
- `countries.csv`
- `clubs.csv`
- `competitions.csv`
- `games.csv`
- `game_events.csv`
- `appearances.csv`
- `transfers.csv`

The full generated data folders are ignored: `raw_data/`, `dataset_clean/`, and `dataset_counter/`.

## Files And Roles

| File or folder | What it does |
| --- | --- |
| `data_install.py` | Downloads raw Kaggle datasets into `raw_data/`. |
| `data_clean.py` | Cleans and filters raw CSVs into benchmark tables under `dataset_clean/`. |
| `counterfact_generation.py` | Builds counterfactual tables from `dataset_clean/` and writes `dataset_counter/`. |
| `extract_sample_dataset.py` | Extracts small player-centered sample tables into `dataset_sample/` or `dataset_counter_sample/`. |
| `extract_question_templates.py` | Parses ignored `raw_questions/level_*` text files into `question_template/level_*.csv`. |
| `question_generation.py` | Generates original or counterfactual question JSON files from templates, seeds, and datasets. |
| `question_template/question_generation/` | Contains the level-specific question construction code. |
| `run_qa_experiment.py` | Samples table rows, prompts a generator model, writes model answers, evaluates them with OpenAI, and writes summaries. |
| `compare_evaluations.py` | Compares `evaluation/` against `counter_evaluation/` and writes a Markdown report of changed outcomes. |
| `export_answer_rows_markdown.py` | Optional utility that materializes answer-supporting rows for generated questions into Markdown. |
| `requirement.txt` | Python dependencies. |
| `.gitignore` | Keeps raw data, generated data, outputs, reports, secrets, and chain-of-thought artifacts out of Git. |

## Supported Models

`run_qa_experiment.py` supports three generator backends.

| Backend | CLI value | Default model | Notes |
| --- | --- | --- | --- |
| Hugging Face causal LM | `--generator-backend llama` | `meta-llama/Llama-3.2-1B-Instruct` | Default backend. The backend name is `llama`, but it can load any compatible `transformers.AutoModelForCausalLM` model ID or local path. |
| OpenAI | `--generator-backend openai` | `gpt-5.4-mini` | Requires `OPENAI_API_KEY`. |
| Gemini | `--generator-backend gemini` | `gemini-3.1-flash-lite` | Requires `GEMINI_API_KEY`. |

Hugging Face aliases supported by `--model` or `--llama-model`:

| Alias | Model ID |
| --- | --- |
| `llama-3.2-1b` | `meta-llama/Llama-3.2-1B` |
| `llama-3.2-1b-instruct` | `meta-llama/Llama-3.2-1B-Instruct` |
| `llama-3.2-3b-instruct` | `meta-llama/Llama-3.2-3B-Instruct` |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B` |
| `llama-3.1-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` |
| `gemma-4-e4b-it` | `google/gemma-4-E4B-it` |
| `gemma-4-e2b-it` | `google/gemma-4-E2B-it` |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` |
| `qwen3.5-2b` | `Qwen/Qwen3.5-2B` |

Gemini aliases:

| Alias | Model ID |
| --- | --- |
| `gemini-3.1-flash-lite` | `gemini-3.1-flash-lite` |

Evaluation is done with OpenAI by default. The evaluator model defaults to `gpt-4o` and can be changed with `--evaluator-model`.

## Pipeline

Install dependencies first:

```bash
pip install -r requirement.txt
```

Set any needed API keys in your shell environment or a local ignored file such as `api.sh` or `.env`. Do not commit secrets.

### 1. Download Raw Data

```bash
python data_install.py
```

Output: `raw_data/`.

This downloads the configured Kaggle datasets and installs their files locally. Kaggle access must be configured for `kagglehub`.

### 2. Clean Data

```bash
python data_clean.py
```

Output: `dataset_clean/`.

This keeps the benchmark columns, normalizes selected tables, filters player-linked tables, and writes the cleaned CSVs.

### 3. Generate Counterfactual Data

```bash
python counterfact_generation.py --seed 42
```

Output: `dataset_counter/`.

This chain-swaps selected attributes in the cleaned dataset so the schema stays the same but table facts change.

### 4. Build Sample Datasets

```bash
python extract_sample_dataset.py --input-dir dataset_clean
python extract_sample_dataset.py --input-dir dataset_counter
```

Outputs:

- `dataset_sample/`
- `dataset_counter_sample/`

These are the small committed examples used to inspect the original and counterfactual table shapes.

### 5. Generate Questions

Original-data questions:

```bash
python question_generation.py --original --level level_1 --level level_2 --level level_3
```

Output: `questions/level_1`, `questions/level_2`, `questions/level_3`.

Counterfactual-data questions:

```bash
python question_generation.py --counter --level level_1 --level level_2 --level level_3
```

Output: `question_counter/level_1`, `question_counter/level_2`, `question_counter/level_3`.

### 6. Run Models And Evaluate

Run one original-data experiment:

```bash
python run_qa_experiment.py \
  --questions-root questions \
  --dataset-dir dataset_clean \
  --results-root results \
  --evaluation-root evaluation \
  --level level_1 \
  --mode all \
  --generator-backend llama \
  --model llama-3.2-1b-instruct
```

Run the matching counterfactual experiment:

```bash
python run_qa_experiment.py \
  --questions-root question_counter \
  --dataset-dir dataset_counter \
  --results-root counter_results \
  --evaluation-root counter_evaluation \
  --level level_1 \
  --mode all \
  --generator-backend llama \
  --model llama-3.2-1b-instruct
```

Useful options:

- `--level` can be passed multiple times.
- `--mode generate` only writes model answers.
- `--mode evaluate` only evaluates existing answers.
- `--mode summarize` only rewrites summary files.
- `--max-questions N` runs a small subset.
- `--dry-run` checks question and table sampling without model calls.
- `--num-rows N` controls the row budget in each prompt.
- `--evaluation-target full` grades the answer plus explanation; `answer` uses answer-only grading.

### 7. Compare Original And Counterfactual Results

```bash
python compare_evaluations.py \
  --evaluation-root evaluation \
  --counter-evaluation-root counter_evaluation \
  --output evaluation_counter_comparison_report.md
```

Output: `evaluation_counter_comparison_report.md`.

The comparison report shows matched question counts, original and counterfactual accuracies, accuracy deltas, changed question IDs, and detailed before/after evaluator rationales.

## Git Policy

Tracked:

- Code
- `README.md`
- `question_template/`
- `dataset_sample/`
- `dataset_counter_sample/`

Ignored:

- Raw and generated full datasets
- Generated questions
- Model outputs and evaluations
- Markdown reports except `README.md`
- API keys and local environment files
- Chain-of-thought artifacts
