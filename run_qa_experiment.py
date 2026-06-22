"""Run counterfactual table QA generation and semantic evaluation experiments.

The default run starts with ``question_counter/level_1``, samples rows from
``dataset_counter/``, generates answers with Llama by default, and grades
the answers with ``gpt-4o``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_GENERATOR_BACKEND = "llama"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_EVALUATOR_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_RANDOM_SEED = 0
GEMINI_MODELS = {
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
}
# The backend is still named "llama" for CLI compatibility, but this registry
# supports any Hugging Face causal LM that works with AutoModelForCausalLM.
LLAMA_MODELS = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-4-e4b-it": "google/gemma-4-E4B-it",
    "gemma-4-e2b-it": "google/gemma-4-E2B-it",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
}
DEFAULT_QUESTIONS_ROOT = Path("question_counter")
DEFAULT_DATASET_DIR = Path("dataset_counter")
DEFAULT_RESULTS_ROOT = Path("counter_results")
DEFAULT_EVALUATION_ROOT = Path("counter_evaluation")
GENERATION_MAX_TOKENS = 128
LLAMA_DEVICE_MAP = "auto"
LLAMA_TRUST_REMOTE_CODE = False

TABLE_FILES = {
    "players": "players.csv",
    "countries": "countries.csv",
    "clubs": "clubs.csv",
    "competitions": "competitions.csv",
    "appearance": "appearances.csv",
    "appearances": "appearances.csv",
    "games": "games.csv",
    "game_events": "game_events.csv",
    "transfers": "transfers.csv",
}


@dataclass(frozen=True)
class TableData:
    """A CSV table loaded with original column and row order preserved."""

    name: str
    csv_file: str
    path: Path
    fieldnames: list[str]
    rows: list[dict[str, str]]


class TableStore:
    """Lazy loader for cleaned CSV tables."""

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self._cache: dict[str, TableData] = {}

    def load(self, table_name: str) -> TableData:
        if table_name in self._cache:
            return self._cache[table_name]

        csv_file = TABLE_FILES.get(table_name, f"{table_name}.csv")
        path = self.dataset_dir / csv_file
        if not path.exists():
            supported = ", ".join(sorted(TABLE_FILES))
            raise FileNotFoundError(
                f"Could not resolve table {table_name!r} to {path}. "
                f"Known table names: {supported}"
            )

        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = list(reader.fieldnames or [])
            rows = [
                {field: row.get(field, "") or "" for field in fieldnames}
                for row in reader
            ]

        table = TableData(
            name=table_name,
            csv_file=csv_file,
            path=path,
            fieldnames=fieldnames,
            rows=rows,
        )
        self._cache[table_name] = table
        return table


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def resolve_questions_path(questions_root: Path, level: str) -> Path:
    candidates = [
        questions_root / level,
        questions_root / f"{level}.json",
        Path(level),
        Path(f"{level}.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find questions for {level!r}. Tried: {tried}")


def load_questions(questions_root: Path, level: str) -> list[dict[str, Any]]:
    path = resolve_questions_path(questions_root, level)
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"Question file must contain a JSON list: {path}")
    return records


def filter_questions(
    records: list[dict[str, Any]],
    question_ids: set[str] | None,
    max_questions: int | None,
) -> list[dict[str, Any]]:
    if question_ids:
        records = [
            record
            for record in records
            if str(record.get("question_id")) in question_ids
        ]
    if max_questions is not None:
        records = records[:max_questions]
    return records


def safe_question_filename(question_id: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(question_id)).strip("._")
    return f"{name or 'question'}.json"


def normalize_model_id(model_name: str, lowercase: bool = False) -> str:
    normalized = re.sub(r"\s+", "-", model_name.strip())
    return normalized.lower() if lowercase else normalized


def resolve_llama_model_id(model_name: str) -> str:
    normalized = normalize_model_id(model_name)
    lookup = normalized.lower()
    for alias, model_id in LLAMA_MODELS.items():
        if lookup in {alias, model_id.lower()}:
            return model_id
    return normalized


def resolve_gemini_model_id(model_name: str) -> str:
    normalized = normalize_model_id(model_name, lowercase=True)
    for alias, model_id in GEMINI_MODELS.items():
        if normalized in {alias, model_id.lower()}:
            return model_id
    return normalized


def safe_model_dir_name(model_name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("._-")
    return name or "model"


def selected_generator_model(args: argparse.Namespace) -> str:
    return args.generator_model


def model_dir_name(args: argparse.Namespace) -> str:
    return safe_model_dir_name(args.model_dir or selected_generator_model(args))


def truth_row_refs(record: dict[str, Any]) -> list[dict[str, Any]]:
    refs = record.get("ground_truth", {}).get("rows", [])
    if not isinstance(refs, list):
        raise ValueError(f"Question {record.get('question_id')} has invalid ground_truth.rows")
    return refs


def unique_tables(record: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    tables: list[str] = []
    for ref in truth_row_refs(record):
        table = str(ref.get("table", "")).strip()
        if table and table not in seen:
            seen.add(table)
            tables.append(table)

    answer_source = record.get("ground_truth", {}).get("answer_source", {})
    if isinstance(answer_source, dict):
        table = str(answer_source.get("table", "")).strip()
        if table and table not in seen:
            tables.append(table)

    if not tables:
        raise ValueError(f"Question {record.get('question_id')} has no referenced tables")
    return tables


def _ref_identity_parts(ref: dict[str, Any]) -> tuple[list[str], list[str]]:
    id_attribute = str(ref.get("id_attribute") or "")
    if not id_attribute:
        raise ValueError("Ground-truth row is missing id_attribute")

    attributes = id_attribute.split("+")
    raw_value = str(ref.get("id_value", ""))
    values = raw_value.split("|") if len(attributes) > 1 else [raw_value]
    if len(attributes) != len(values):
        raise ValueError(
            f"Composite id {id_attribute!r} has {len(attributes)} attributes but "
            f"id_value {raw_value!r} has {len(values)} parts"
        )
    return attributes, values


def find_truth_indices(table: TableData, refs: list[dict[str, Any]]) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    lookup_cache: dict[tuple[str, ...], dict[tuple[str, ...], list[int]]] = {}

    for ref in refs:
        attributes, values = _ref_identity_parts(ref)
        attribute_key = tuple(attributes)
        value_key = tuple(values)

        if attribute_key not in lookup_cache:
            missing = [
                attribute
                for attribute in attributes
                if attribute not in table.fieldnames
            ]
            if missing:
                raise KeyError(
                    f"{table.name}: id attribute(s) {missing!r} are not in {table.csv_file}"
                )

            lookup: dict[tuple[str, ...], list[int]] = {}
            for index, row in enumerate(table.rows):
                row_key = tuple(str(row.get(attribute, "")) for attribute in attributes)
                lookup.setdefault(row_key, []).append(index)
            lookup_cache[attribute_key] = lookup

        matches = lookup_cache[attribute_key].get(value_key, [])
        if not matches:
            condition = " and ".join(
                f"{attribute} == {value!r}"
                for attribute, value in zip(attributes, values)
            )
            raise ValueError(f"{table.name}: no row where {condition}")
        for index in matches:
            if index not in seen:
                seen.add(index)
                indices.append(index)
    return indices


def choose_distractors(
    table: TableData,
    truth_indices: set[int],
    count: int,
    policy: str,
    rng: random.Random,
) -> list[int]:
    if count <= 0:
        return []

    candidates = [
        index
        for index in range(len(table.rows))
        if index not in truth_indices
    ]
    if policy == "random":
        rng.shuffle(candidates)
    return candidates[:count]


def sample_tables_for_question(
    record: dict[str, Any],
    table_store: TableStore,
    total_rows: int,
    distractor_policy: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    tables = unique_tables(record)
    assigned_rows_per_table = max(1, total_rows // len(tables))
    refs_by_table: dict[str, list[dict[str, Any]]] = {table: [] for table in tables}
    for ref in truth_row_refs(record):
        table = str(ref.get("table", "")).strip()
        refs_by_table.setdefault(table, []).append(ref)

    sampled_tables: list[dict[str, Any]] = []
    for table_name in tables:
        table = table_store.load(table_name)
        table_refs = refs_by_table.get(table_name, [])
        truth_indices = find_truth_indices(table, table_refs) if table_refs else []
        truth_set = set(truth_indices)
        effective_row_budget = max(assigned_rows_per_table, len(truth_indices))

        if len(table.rows) <= effective_row_budget:
            selected_indices = list(range(len(table.rows)))
        else:
            distractor_count = max(0, effective_row_budget - len(truth_indices))
            distractors = choose_distractors(
                table=table,
                truth_indices=truth_set,
                count=distractor_count,
                policy=distractor_policy,
                rng=rng,
            )
            selected_indices = sorted(set(truth_indices + distractors))

        sampled_rows = [
            {
                "row_index": index,
                "csv_row_number": index + 2,
                "is_ground_truth": index in truth_set,
                "values": {
                    field: table.rows[index].get(field, "")
                    for field in table.fieldnames
                },
            }
            for index in selected_indices
        ]

        sampled_tables.append(
            {
                "table": table.name,
                "csv_file": table.csv_file,
                "csv_path": str(table.path),
                "columns": table.fieldnames,
                "row_budget": effective_row_budget,
                "assigned_row_budget": assigned_rows_per_table,
                "effective_row_budget": effective_row_budget,
                "row_budget_overridden_by_ground_truth": (
                    effective_row_budget > assigned_rows_per_table
                ),
                "total_csv_rows": len(table.rows),
                "truth_row_count": len(truth_indices),
                "truth_row_indices": truth_indices,
                "truth_csv_row_numbers": [index + 2 for index in truth_indices],
                "selected_row_indices": selected_indices,
                "selected_csv_row_numbers": [index + 2 for index in selected_indices],
                "rows": sampled_rows,
            }
        )

    return sampled_tables


def rows_to_csv(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return buffer.getvalue().rstrip("\n")


def build_gold_full_answer(answer: str, explanation: str) -> str:
    return f"Answer: {answer}. Explanation: {explanation}"


def is_null_reference_explanation(explanation: object) -> bool:
    text = str(explanation or "").strip().lower()
    return text in {"", "null", "none", "nan", "n/a", "na"}


def build_prompt(record: dict[str, Any], sampled_tables: list[dict[str, Any]]) -> str:
    parts = [
        "You will be answering a table question.",
        "Use only the CSV tables below.",
        "The excerpts preserve the source column order and selected row order.",
        "Return the final answer plus a brief explanation grounded in the shown table values.",
        "Use this format: Answer: <short final answer>. Explanation: <brief table-grounded reason>.",
        "",
        f"Question: {record['question']}",
        "",
    ]
    for sampled in sampled_tables:
        rows = [row["values"] for row in sampled["rows"]]
        parts.extend(
            [
                f"Table: {sampled['table']} ({sampled['csv_file']})",
                "```csv",
                rows_to_csv(sampled["columns"], rows),
                "```",
                "",
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


class OpenAIGenerator:
    """OpenAI Chat Completions wrapper for GPT text generation."""

    def __init__(self, model_name: str, temperature: float, seed: int) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI generation requires the openai package. "
                "Install it with: pip install -r requirement.txt"
            ) from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before running OpenAI generation.")

        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_completion_tokens=GENERATION_MAX_TOKENS,
            temperature=self.temperature,
            seed=self.seed,
            messages=[
                {
                    "role": "system",
                    "content": "You answer table questions with concise factual answers and brief table-grounded explanations.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()


class GeminiGenerator:
    def __init__(self, model_name: str, temperature: float) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini generation requires the google-genai package. "
                "Install it with: pip install -r requirement.txt"
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY before running Gemini generation.")

        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                max_output_tokens=GENERATION_MAX_TOKENS,
                temperature=self.temperature,
                system_instruction=(
                    "You answer table questions with concise factual answers "
                    "and brief table-grounded explanations."
                ),
            ),
        )
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        return "".join(str(getattr(part, "text", "") or "") for part in parts).strip()


class LlamaGenerator:
    """Hugging Face Transformers wrapper for causal LM text generation."""

    def __init__(self, model_name: str, seed: int) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face generation requires torch and transformers. "
                "Install them with: pip install -r requirement.txt"
            ) from exc

        self.torch = torch
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=LLAMA_TRUST_REMOTE_CODE,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=LLAMA_DEVICE_MAP,
            trust_remote_code=LLAMA_TRUST_REMOTE_CODE,
        )
        self.disable_qwen_thinking = model_name.lower().startswith("qwen/qwen3.5")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You answer table questions with concise factual answers and brief table-grounded explanations.",
            },
            {"role": "user", "content": prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            chat_template_kwargs: dict[str, Any] = {}
            if self.disable_qwen_thinking:
                chat_template_kwargs["enable_thinking"] = False
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        else:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]

        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": GENERATION_MAX_TOKENS,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": False,
        }

        with self.torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)
        answer_ids = output_ids[0][input_length:]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def build_generator(args: argparse.Namespace) -> Any:
    load_dotenv_if_available()
    if args.generator_backend == "openai":
        return OpenAIGenerator(
            model_name=args.generator_model,
            temperature=args.temperature,
            seed=args.seed,
        )
    if args.generator_backend == "gemini":
        return GeminiGenerator(
            model_name=args.generator_model,
            temperature=args.temperature,
        )
    if args.generator_backend == "llama":
        return LlamaGenerator(model_name=args.generator_model, seed=args.seed)
    raise ValueError(f"Unsupported generator backend: {args.generator_backend}")

def run_generation(args: argparse.Namespace, level: str, records: list[dict[str, Any]]) -> None:
    model_dir = model_dir_name(args)
    output_dir = args.results_root / model_dir / level
    table_store = TableStore(args.dataset_dir)
    rng = random.Random(args.seed)

    if args.dry_run:
        for record in records:
            sampled = sample_tables_for_question(
                record=record,
                table_store=table_store,
                total_rows=args.num_rows,
                distractor_policy=args.distractor_policy,
                rng=rng,
            )
            counts = ", ".join(
                f"{table['table']}={len(table['rows'])}" for table in sampled
            )
            print(f"[dry-run] {level} q{record.get('question_id')}: {counts}")
        return

    generator = build_generator(args)

    for index, record in enumerate(records, start=1):
        question_id = record.get("question_id")
        output_path = output_dir / safe_question_filename(question_id)
        if output_path.exists() and not args.overwrite_results:
            print(f"[generation] skip existing {output_path}")
            continue

        sampled = sample_tables_for_question(
            record=record,
            table_store=table_store,
            total_rows=args.num_rows,
            distractor_policy=args.distractor_policy,
            rng=rng,
        )
        assigned_rows_per_table = max(1, args.num_rows // len(sampled))
        prompt = build_prompt(record, sampled)
        model_answer = generator.generate(prompt)
        reference_answer = str(record.get("answer") or "")
        reference_explanation = str(record.get("explanation") or "")
        payload = {
            "level": level,
            "question_id": question_id,
            "template_question_id": record.get("template_question_id"),
            "question": record.get("question"),
            "reference_answer": reference_answer,
            "reference_explanation": reference_explanation,
            "reference_full_answer": build_gold_full_answer(
                reference_answer,
                reference_explanation,
            ),
            "ground_truth": record.get("ground_truth"),
            "sampling": {
                "total_row_budget": args.num_rows,
                "rows_per_table": assigned_rows_per_table,
                "assigned_rows_per_table": assigned_rows_per_table,
                "effective_rows_per_table": {
                    table["table"]: table["effective_row_budget"]
                    for table in sampled
                },
                "row_budget_overridden_by_ground_truth": any(
                    table["row_budget_overridden_by_ground_truth"]
                    for table in sampled
                ),
                "distractor_policy": args.distractor_policy,
                "seed": args.seed,
                "tables": sampled,
            },
            "prompt": prompt,
            "generator_backend": args.generator_backend,
            "generator_model": args.generator_model,
            "temperature": args.temperature,
            "seed": args.seed,
            "model_dir": model_dir,
            "model_answer": model_answer,
        }
        write_json(output_path, payload)
        print(f"[generation] {level} {index}/{len(records)} -> {output_path}")


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError(f"Expected evaluator JSON object, got: {type(value).__name__}")
    return value


def evaluate_with_openai(
    client: Any,
    evaluator_model: str,
    question: str,
    reference_answer: str,
    model_answer: str,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    rubric = (
        "You are grading answer equivalence for a table QA benchmark. "
        "Use only the provided reference_answer as the ground truth; do not use "
        "outside knowledge, original-world facts, or your own knowledge of the "
        "question subject. The model answer is correct when its main answer "
        "matches the reference_answer semantically. Extra or irrelevant "
        "explanatory information is allowed as long as it does not contradict "
        "the main answer. For Yes/No reference answers, grade the primary "
        "Yes/No stance first: if the model answer has the same main Yes or No, "
        "mark it correct even when the explanation gives extra table details "
        "or the underlying value that caused that Yes/No answer. Do not treat "
        "an explanation as contradictory unless it clearly reverses the main "
        "Yes/No answer or asserts an incompatible answer. If reference_answer is NULL, accept answers that "
        "state NULL, no listed value, not available, not provided, or otherwise "
        "indicate the table/reference has no answer. Do not mark an answer "
        "incorrect merely because it explains NULL in words. Ignore harmless "
        "casing, punctuation, formatting, date formatting, units, or wording "
        "differences. Return only JSON with keys: correct (boolean), accuracy "
        "(0 or 1), rationale (short string)."
    )
    user_payload = {
        "reference_answer": reference_answer,
        "model_answer": model_answer,
    }
    response = client.chat.completions.create(
        model=evaluator_model,
        response_format={"type": "json_object"},
        temperature=temperature,
        seed=seed,
        messages=[
            {"role": "system", "content": rubric},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = parse_json_object(content)
    correct = bool(parsed.get("correct", parsed.get("accuracy", 0)))
    accuracy = 1 if correct else 0
    return {
        "correct": correct,
        "accuracy": accuracy,
        "rationale": str(parsed.get("rationale", "")),
        "raw_evaluator_response": parsed,
    }


def evaluate_full_answer_with_openai(
    client: Any,
    evaluator_model: str,
    question: str,
    reference_answer: str,
    reference_explanation: str,
    reference_explanation_is_null: bool,
    model_response: str,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    rubric = """
    You are grading a table QA benchmark response.

The tables may contain original or counterfactual facts. Treat reference_answer as the authoritative ground truth, even if it conflicts with real-world knowledge. Do not use your own knowledge to judge the answer.

Return only valid JSON:
{
"answer_correct": boolean,
"grounded": boolean,
"accuracy": 0 or 1,
"grounded_accuracy": 0 or 1,
"rationale": "short explanation"
}

Grading rules:

1. First grade the model's main/final answer against reference_answer.

   * If the final answer matches reference_answer, set answer_correct=true and accuracy=1.
   * If the final answer is wrong, missing, ambiguous, or contradicted by another final-answer claim, set answer_correct=false and accuracy=0.
   * If the final answer is wrong, it is incorrect even if the explanation seems to imply the right answer.

2. Ignore harmless differences in casing, punctuation, formatting, wording, date format, units, or equivalent aliases.

3. Use reference_explanation only to check whether the model's explanation is compatible with the table evidence. The model does not need to follow the exact same reasoning path.

4. Set grounded=true when:

   * the final answer is correct and the explanation is absent, vague, or compatible with reference_explanation; or
   * the model uses a different but valid reasoning path.

5. Set grounded=false when:

   * the final answer is correct, but the explanation gives central table facts that contradict reference_explanation, such as wrong country pairs, continent pairs, club mappings, transfer pairs, dates, counts, or joined entities.
   * This is especially important for counterfactual data: if the model gets the right final answer but explains it using original-world or original-table facts, grounded=false.

6. grounded_accuracy = 1 only when both answer_correct=true and grounded=true. Otherwise grounded_accuracy=0.

Examples:

* Final answer wrong, explanation implies correct answer → answer_correct=false, grounded_accuracy=0.
* Final answer correct, no explanation → answer_correct=true, grounded=true.
* Final answer correct, different compatible reasoning path → answer_correct=true, grounded=true.
* Final answer correct, explanation uses contradictory table facts → answer_correct=true, grounded=false.

    """
    user_payload = {
        "question": question,
        "reference_answer": reference_answer,
        "reference_explanation": reference_explanation,
        "reference_explanation_is_null": reference_explanation_is_null,
        "model_response": model_response,
    }
    response = client.chat.completions.create(
        model=evaluator_model,
        response_format={"type": "json_object"},
        temperature=temperature,
        seed=seed,
        messages=[
            {"role": "system", "content": rubric},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = parse_json_object(content)
    correct = bool(parsed.get("correct", parsed.get("accuracy", 0)))
    accuracy = 1 if correct else 0
    return {
        "correct": correct,
        "accuracy": accuracy,
        "rationale": str(parsed.get("rationale", "")),
        "raw_evaluator_response": parsed,
    }


def run_evaluation(args: argparse.Namespace, level: str, records: list[dict[str, Any]]) -> None:
    load_dotenv_if_available()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running evaluation.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI evaluation requires the openai package. "
            "Install it with: pip install -r requirement.txt"
        ) from exc

    client = OpenAI()
    model_dir = model_dir_name(args)
    result_dir = args.results_root / model_dir / level
    evaluation_dir = args.evaluation_root / model_dir / level
    records_by_id = {str(record.get("question_id")): record for record in records}

    for index, record in enumerate(records, start=1):
        question_id = record.get("question_id")
        result_path = result_dir / safe_question_filename(question_id)
        if not result_path.exists():
            print(f"[evaluation] missing result for q{question_id}: {result_path}")
            continue

        evaluation_path = evaluation_dir / safe_question_filename(question_id)
        if evaluation_path.exists() and not args.overwrite_evaluations:
            print(f"[evaluation] skip existing {evaluation_path}")
            continue

        result = load_json(result_path)
        question_record = records_by_id.get(str(question_id), record)
        question = str(result.get("question") or question_record.get("question") or "")
        reference_answer = str(
            result.get("reference_answer")
            or question_record.get("answer")
            or ""
        )
        reference_explanation_value = result.get(
            "reference_explanation",
            question_record.get("explanation"),
        )
        reference_explanation = str(reference_explanation_value or "")
        reference_explanation_is_null = is_null_reference_explanation(
            reference_explanation_value
        )
        gold_full_answer = str(
            result.get("reference_full_answer")
            or build_gold_full_answer(reference_answer, reference_explanation)
        )
        model_answer = str(result.get("model_answer") or "")

        if args.evaluation_target == "answer":
            grade = evaluate_with_openai(
                client=client,
                evaluator_model=args.evaluator_model,
                question=question,
                reference_answer=reference_answer,
                model_answer=model_answer,
                temperature=args.temperature,
                seed=args.seed,
            )
        else:
            grade = evaluate_full_answer_with_openai(
                client=client,
                evaluator_model=args.evaluator_model,
                question=question,
                reference_answer=reference_answer,
                reference_explanation=reference_explanation,
                reference_explanation_is_null=reference_explanation_is_null,
                model_response=model_answer,
                temperature=args.temperature,
                seed=args.seed,
            )
        payload = {
            "level": level,
            "question_id": question_id,
            "question": question,
            "reference_answer": reference_answer,
            "reference_explanation": reference_explanation,
            "reference_explanation_is_null": reference_explanation_is_null,
            "gold_full_answer": gold_full_answer,
            "model_response": model_answer,
            "model_answer": model_answer,
            "evaluation_target": args.evaluation_target,
            "result_file": str(result_path),
            "generator_backend": result.get("generator_backend"),
            "generator_model": result.get("generator_model"),
            "model_dir": model_dir,
            "evaluator_model": args.evaluator_model,
            "temperature": args.temperature,
            "seed": args.seed,
            **grade,
        }
        write_json(evaluation_path, payload)
        print(f"[evaluation] {level} {index}/{len(records)} -> {evaluation_path}")


def summarize_level(
    evaluation_root: Path,
    model_dir: str,
    level: str,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    evaluation_dir = evaluation_root / model_dir / level
    files = sorted(evaluation_dir.glob("*.json"))
    selected_question_ids = (
        {str(record.get("question_id")) for record in records}
        if records is not None
        else None
    )
    accuracies: list[float] = []
    stale_files_skipped = 0
    for path in files:
        payload = load_json(path)
        question_id = str(payload.get("question_id", path.stem))
        if selected_question_ids is not None and question_id not in selected_question_ids:
            stale_files_skipped += 1
            continue
        if "accuracy" not in payload:
            continue
        accuracies.append(float(payload["accuracy"]))

    average = sum(accuracies) / len(accuracies) if accuracies else None
    summary = {
        "level": level,
        "model_dir": model_dir,
        "evaluation_dir": str(evaluation_dir),
        "num_selected_questions": len(selected_question_ids) if selected_question_ids is not None else None,
        "num_evaluated": len(accuracies),
        "num_stale_files_skipped": stale_files_skipped,
        "average_accuracy": average,
    }
    write_json(evaluation_root / model_dir / f"{level}_summary.json", summary)
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selectable table-QA generation and GPT semantic evaluation."
    )
    parser.add_argument(
        "--level",
        action="append",
        default=None,
        help="Question level to run. Can be passed multiple times. Defaults to level_1.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "generate", "evaluate", "summarize"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument("--questions-root", type=Path, default=DEFAULT_QUESTIONS_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument(
        "--num-rows",
        type=int,
        default=20,
        help="Total row budget n for each question prompt.",
    )
    parser.add_argument(
        "--distractor-policy",
        choices=["first", "random"],
        default="first",
        help="How to choose non-ground-truth rows before restoring CSV order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Fixed random seed for row sampling and model calls that support seeds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for OpenAI generation/evaluation calls. Defaults to 0.",
    )
    parser.add_argument(
        "--generator-backend",
        "--generator",
        choices=["openai", "gemini", "llama"],
        default=None,
        help="Generation backend. Defaults to llama.",
    )
    parser.add_argument(
        "--model",
        "--generator-model",
        dest="generator_model",
        default=None,
        help=(
            "Model id for answer generation. Defaults to "
            f"{DEFAULT_LLAMA_MODEL} for llama, {DEFAULT_GEMINI_MODEL} for gemini, "
            f"or {DEFAULT_OPENAI_MODEL} for openai. "
            "Gemini aliases: "
            + ", ".join(f"{alias}={model_id}" for alias, model_id in GEMINI_MODELS.items())
            + ". "
            "Hugging Face aliases: "
            + ", ".join(f"{alias}={model_id}" for alias, model_id in LLAMA_MODELS.items())
            + "."
        ),
    )
    parser.add_argument("--gpt-model", dest="generator_model", help=argparse.SUPPRESS)
    parser.add_argument(
        "--llama-model",
        default=None,
        help="Hugging Face model id or local path for the llama/HF backend. Passing this without --model selects the llama backend.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory name to use under results/ and evaluation/. Defaults to the model id slug.",
    )
    parser.add_argument("--evaluator-model", default=DEFAULT_EVALUATOR_MODEL)
    parser.add_argument(
        "--evaluation-target",
        choices=["full", "answer"],
        default="full",
        help="Evaluate the combined answer-plus-explanation gold target, or the legacy answer-only target.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=None,
        help="Only run one question id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Only run the first N questions after question-id filtering.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Regenerate result files that already exist.",
    )
    parser.add_argument(
        "--overwrite-evaluations",
        action="store_true",
        help="Regenerate evaluation files that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate question/table sampling without calling a generator or OpenAI evaluator.",
    )
    args = parser.parse_args(argv)
    if args.generator_backend is None:
        args.generator_backend = (
            "llama" if args.llama_model and args.generator_model is None else DEFAULT_GENERATOR_BACKEND
        )
    if args.generator_model is None:
        if args.generator_backend == "llama":
            args.generator_model = args.llama_model or DEFAULT_LLAMA_MODEL
        elif args.generator_backend == "gemini":
            args.generator_model = DEFAULT_GEMINI_MODEL
        else:
            args.generator_model = DEFAULT_OPENAI_MODEL
    args.generator_model = normalize_model_id(
        args.generator_model,
        lowercase=args.generator_backend in {"gemini", "openai"},
    )
    if args.generator_backend == "llama":
        args.generator_model = resolve_llama_model_id(args.generator_model)
    elif args.generator_backend == "gemini":
        args.generator_model = resolve_gemini_model_id(args.generator_model)
    if args.num_rows <= 0:
        parser.error("--num-rows must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.max_questions is not None and args.max_questions <= 0:
        parser.error("--max-questions must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    levels = args.level or ["level_1"]
    question_ids = set(args.question_id) if args.question_id else None

    for level in levels:
        records = load_questions(args.questions_root, level)
        records = filter_questions(records, question_ids, args.max_questions)
        if not records:
            print(f"[{level}] no questions selected")
            continue

        if args.mode in {"all", "generate"}:
            run_generation(args, level, records)

        if args.mode in {"all", "evaluate"} and not args.dry_run:
            run_evaluation(args, level, records)

        if args.mode in {"all", "evaluate", "summarize"} and not args.dry_run:
            model_dir = model_dir_name(args)
            summary = summarize_level(args.evaluation_root, model_dir, level, records)
            average = summary["average_accuracy"]
            if average is None:
                print(f"[summary] {model_dir}/{level}: no evaluation files with accuracy")
            else:
                print(
                    f"[summary] {model_dir}/{level}: average accuracy "
                    f"{average:.4f} over {summary['num_evaluated']} questions"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
