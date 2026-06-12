from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# Change these if your provider account exposes different public model IDs.
OPENAI_ANSWER_MODEL = "gpt-5.5"
GEMINI_ANSWER_MODEL = "gemini-3.5-flash"
OPENAI_EVAL_MODEL = "gpt-4o-mini"

BASE_DIR = Path(__file__).resolve().parent
REQUIRED_QUESTION_COLUMNS = {"question_id", "question", "answer"}

RESULT_FIELDS = [
    "level",
    "question_type",
    "question_id",
    "question",
    "gold_answer",
    "table_files",
    "prompt",
    "answer_model",
    "generated_answer",
    "eval_model",
    "eval_correct",
    "eval_reason",
    "answer_error",
    "eval_error",
]

EVALUATION_FIELDS = [
    "level",
    "question_type",
    "question_id",
    "question",
    "gold_answer",
    "answer_model",
    "generated_answer",
    "eval_model",
    "eval_correct",
    "eval_reason",
    "answer_error",
    "eval_error",
]


@dataclass(frozen=True)
class BenchmarkTask:
    level: str
    question_type: str
    question_file: Path
    table_files: tuple[Path, ...]
    answer_file: Path


@dataclass(frozen=True)
class AnswerModel:
    key: str
    provider: str
    model_name: str


TASKS: tuple[BenchmarkTask, ...] = (
    BenchmarkTask(
        level="1",
        question_type="capital_city",
        question_file=BASE_DIR / "questions/level_1/capital_city.csv",
        table_files=(BASE_DIR / "data/level_1/countries_change.csv",),
        answer_file=BASE_DIR / "data/level_1/countries_answer.csv",
    ),
    BenchmarkTask(
        level="2",
        question_type="player_age",
        question_file=BASE_DIR / "questions/level_2/player_age.csv",
        table_files=(BASE_DIR / "data/level_2/players_change.csv",),
        answer_file=BASE_DIR / "data/level_2/players_answer.csv",
    ),
    BenchmarkTask(
        level="3",
        question_type="club_country",
        question_file=BASE_DIR / "questions/level_3/club_country.csv",
        table_files=(
            BASE_DIR / "data/level_3/clubs_change.csv",
            BASE_DIR / "data/level_3/countries.csv",
        ),
        answer_file=BASE_DIR / "data/level_3/clubs_answer.csv",
    ),
    BenchmarkTask(
        level="4",
        question_type="candidate_teammate",
        question_file=BASE_DIR / "questions/level_4/candidate_teammate.csv",
        table_files=(BASE_DIR / "data/level_4/transfer_teammate.csv",),
        answer_file=BASE_DIR / "data/level_4/transfer_teammate_answer.csv",
    ),
    BenchmarkTask(
        level="4",
        question_type="non_overlapping_teammate",
        question_file=BASE_DIR / "questions/level_4/non_overlapping_teammate.csv",
        table_files=(BASE_DIR / "data/level_4/transfer_nonoverlap.csv",),
        answer_file=BASE_DIR / "data/level_4/transfer_nonoverlap_answer.csv",
    ),
    BenchmarkTask(
        level="4",
        question_type="transfer_path",
        question_file=BASE_DIR / "questions/level_4/transfer_path.csv",
        table_files=(BASE_DIR / "data/level_4/transfer_path.csv",),
        answer_file=BASE_DIR / "data/level_4/transfer_path_answer.csv",
    ),
)

ANSWER_MODELS: dict[str, AnswerModel] = {
    "gpt55": AnswerModel("gpt55", "openai", OPENAI_ANSWER_MODEL),
    "gemini35flash": AnswerModel("gemini35flash", "gemini", GEMINI_ANSWER_MODEL),
}


def load_dotenv_if_available() -> None:
    """Support a local .env file while still using environment variables."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(BASE_DIR / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM answer generation and automatic evaluation for the table benchmark."
    )
    parser.add_argument("--level", required=True, choices=["1", "2", "3", "4", "all"])
    parser.add_argument(
        "--question_type",
        help="Optional question type. If omitted, all question types for the level are run.",
    )
    parser.add_argument("--num_questions", default="all", help="Integer count or 'all'.")
    parser.add_argument("--model", default="all", choices=["gpt55", "gemini35flash", "all"])
    parser.add_argument(
        "--eval",
        dest="run_eval",
        action="store_true",
        help="Run GPT-4o-mini evaluation after answer generation.",
    )
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing result text files instead of starting fresh for this run.",
    )
    return parser.parse_args()


def parse_num_questions(value: str) -> int | None:
    if value == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("--num_questions must be an integer or 'all'.") from exc
    if parsed <= 0:
        raise ValueError("--num_questions must be positive or 'all'.")
    return parsed


def select_tasks(level: str, question_type: str | None) -> list[BenchmarkTask]:
    selected = [
        task
        for task in TASKS
        if (level == "all" or task.level == level)
        and (question_type is None or task.question_type == question_type)
    ]
    if not selected:
        available = sorted(
            {task.question_type for task in TASKS if level == "all" or task.level == level}
        )
        raise ValueError(
            f"No tasks matched level={level!r}, question_type={question_type!r}. "
            f"Available question types: {', '.join(available)}"
        )
    return selected


def select_models(model_key: str) -> list[AnswerModel]:
    if model_key == "all":
        return list(ANSWER_MODELS.values())
    return [ANSWER_MODELS[model_key]]


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def validate_task(task: BenchmarkTask) -> None:
    paths = [task.question_file, *task.table_files]
    missing = [relative_path(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")

    if not task.answer_file.exists():
        print(f"Warning: answer/debug file is missing: {relative_path(task.answer_file)}")

    question_df = pd.read_csv(task.question_file, nrows=0)
    missing_columns = REQUIRED_QUESTION_COLUMNS - set(question_df.columns)
    if missing_columns:
        raise ValueError(
            f"{relative_path(task.question_file)} is missing required column(s): "
            f"{', '.join(sorted(missing_columns))}"
        )


def load_questions(task: BenchmarkTask, num_questions: int | None) -> pd.DataFrame:
    df = pd.read_csv(task.question_file, dtype=str).fillna("")
    missing_columns = REQUIRED_QUESTION_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{relative_path(task.question_file)} is missing required column(s): "
            f"{', '.join(sorted(missing_columns))}"
        )
    if num_questions is not None:
        df = df.head(num_questions)
    return df


def read_csv_text(path: Path) -> str:
    # Keep the file as CSV text; only trim final newlines so the prompt layout is stable.
    return path.read_text(encoding="utf-8").rstrip("\n")


def build_answer_prompt(task: BenchmarkTask, question: str) -> str:
    if task.level == "3":
        csv_1 = read_csv_text(task.table_files[0])
        csv_2 = read_csv_text(task.table_files[1])
        return (
            "You will be given tables in csv format. You will be asked a question that can "
            "be answered by the tables. Use the information only in the tables.\n\n"
            f"Table 1:\n{csv_1}\n\n"
            f"Table 2:\n{csv_2}\n\n\n"
            f"Question:\n{question}\n\n"
            "Answer:\n"
        )

    csv_text = read_csv_text(task.table_files[0])
    return (
        "You will be given a table in csv format. You will be asked a question that can "
        "be answered by the table. Use the information only in the table.\n\n"
        f"Table:\n{csv_text}\n\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )


def build_eval_prompt(question: str, gold_answer: str, model_answer: str) -> str:
    return (
        "You are evaluating an answer for a table-based question.\n\n"
        "Use only the question, gold answer, and model answer presented below. "
        "Do not use your own outside knowledge or real-world facts. If the gold answer "
        "conflicts with what you know, follow the gold answer provided here.\n\n"
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold_answer}\n\n"
        f"Model answer:\n{model_answer}\n\n"
        "Decide whether the model answer is correct.\n\n"
        "The model answer does not need to match the gold answer exactly. It is correct if "
        "the main content is correct.\n\n"
        "For list answers, the answer should contain the correct items. If order matters, "
        "such as transfer path, the order should be correct.\n\n"
        "Return only a JSON object with the following fields:\n"
        "{\n"
        '  "correct": true or false,\n'
        '  "reason": "short explanation"\n'
        "}\n"
    )


def get_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set.")
    return value


def safe_error(exc: BaseException) -> str:
    message = str(exc)
    for env_name in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        secret = os.getenv(env_name, "")
        if secret and len(secret) >= 8:
            message = message.replace(secret, f"<{env_name}>")
    return f"{type(exc).__name__}: {message}"


def extract_openai_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    parts: list[str] = []
    output = getattr(response, "output", None)
    if output:
        for item in output:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(str(text))

    choices = getattr(response, "choices", None)
    if choices:
        for choice in choices:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message else None
            if content:
                parts.append(str(content))

    if parts:
        return "\n".join(parts).strip()
    raise RuntimeError("OpenAI response did not contain extractable text.")


def call_openai(prompt: str, model_name: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

    responses_api = getattr(client, "responses", None)
    if responses_api is not None and hasattr(responses_api, "create"):
        response = responses_api.create(model=model_name, input=prompt)
        return extract_openai_text(response)

    # Fallback for older openai SDKs that only expose Chat Completions.
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_openai_text(response)


def call_gemini(prompt: str, model_name: str) -> str:
    from google import genai

    client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))
    response = client.models.generate_content(model=model_name, contents=prompt)
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    raise RuntimeError("Gemini response did not contain extractable text.")


def call_answer_model(prompt: str, answer_model: AnswerModel) -> str:
    if answer_model.provider == "openai":
        return call_openai(prompt, answer_model.model_name)
    if answer_model.provider == "gemini":
        return call_gemini(prompt, answer_model.model_name)
    raise ValueError(f"Unsupported provider: {answer_model.provider}")


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def parse_eval_json(text: str) -> tuple[bool | None, str, str]:
    """Return (correct, reason, parse_error). Raw text is kept as reason on failure."""
    candidates = [text.strip(), strip_json_fence(text)]

    fenced_or_raw = strip_json_fence(text)
    start = fenced_or_raw.find("{")
    end = fenced_or_raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(fenced_or_raw[start : end + 1])

    last_error = ""
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"JSONDecodeError: {exc}"
            continue

        correct = parsed.get("correct")
        if isinstance(correct, str):
            lowered = correct.strip().lower()
            if lowered == "true":
                correct = True
            elif lowered == "false":
                correct = False
            else:
                correct = None
        elif not isinstance(correct, bool):
            correct = None

        reason = str(parsed.get("reason", "")).strip()
        return correct, reason, ""

    return None, text.strip(), last_error or "Could not parse evaluator JSON."


def evaluate_answer(question: str, gold_answer: str, generated_answer: str) -> tuple[Any, str, str]:
    prompt = build_eval_prompt(question, gold_answer, generated_answer)
    eval_text = call_openai(prompt, OPENAI_EVAL_MODEL)
    correct, reason, parse_error = parse_eval_json(eval_text)
    eval_correct: Any = "" if correct is None else correct
    return eval_correct, reason, parse_error


def output_dir_from_arg(value: str) -> Path:
    output_dir = Path(value)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    return output_dir


def task_output_path(output_dir: Path, task: BenchmarkTask, answer_model: AnswerModel) -> Path:
    return output_dir / f"level_{task.level}_{task.question_type}_{answer_model.key}.txt"


def initialize_text_file(path: Path, append: bool, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        return
    path.write_text(f"{title}\n\n", encoding="utf-8")


def format_text_value(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return ""
    return str(value)


def append_text_record(
    path: Path,
    record_type: str,
    row: dict[str, Any],
    fields: list[str],
) -> None:
    lines = [f"========== {record_type} =========="]
    for field in fields:
        value = format_text_value(row.get(field, ""))
        if "\n" in value:
            lines.append(f"{field}:")
            lines.append(value)
        else:
            lines.append(f"{field}: {value}")
    lines.append("")

    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def make_empty_row(
    task: BenchmarkTask,
    question_row: pd.Series,
    prompt: str,
    answer_model: AnswerModel,
    run_eval: bool,
) -> dict[str, Any]:
    return {
        "level": task.level,
        "question_type": task.question_type,
        "question_id": question_row["question_id"],
        "question": question_row["question"],
        "gold_answer": question_row["answer"],
        "table_files": ";".join(relative_path(path) for path in task.table_files),
        "prompt": prompt,
        "answer_model": answer_model.model_name,
        "generated_answer": "",
        "eval_model": OPENAI_EVAL_MODEL if run_eval else "",
        "eval_correct": "",
        "eval_reason": "",
        "answer_error": "",
        "eval_error": "",
    }


def run_task(
    task: BenchmarkTask,
    answer_model: AnswerModel,
    num_questions: int | None,
    output_dir: Path,
    evaluation_output_path: Path | None,
    run_eval: bool,
) -> None:
    questions = load_questions(task, num_questions)
    output_path = task_output_path(output_dir, task, answer_model)

    print(
        f"Running level {task.level} / {task.question_type} with "
        f"{answer_model.model_name} on {len(questions)} question(s)."
    )

    for idx, question_row in questions.iterrows():
        question_number = idx + 1
        question = question_row["question"]
        prompt = build_answer_prompt(task, question)
        row = make_empty_row(task, question_row, prompt, answer_model, run_eval)

        print(
            f"[level {task.level} {task.question_type} {answer_model.key}] "
            f"{question_number}/{len(questions)} question_id={question_row['question_id']}"
        )

        try:
            generated_answer = call_answer_model(prompt, answer_model)
            row["generated_answer"] = generated_answer
        except Exception as exc:  # Continue if one provider call fails.
            row["answer_error"] = safe_error(exc)

        if run_eval and row["generated_answer"] and not row["answer_error"]:
            try:
                eval_correct, eval_reason, eval_error = evaluate_answer(
                    question=question,
                    gold_answer=question_row["answer"],
                    generated_answer=row["generated_answer"],
                )
                row["eval_correct"] = eval_correct
                row["eval_reason"] = eval_reason
                row["eval_error"] = eval_error
            except Exception as exc:
                row["eval_error"] = safe_error(exc)
        elif run_eval and row["answer_error"]:
            row["eval_error"] = "Skipped evaluation because answer generation failed."

        append_text_record(output_path, "RESULT", row, RESULT_FIELDS)
        if run_eval and evaluation_output_path is not None:
            append_text_record(evaluation_output_path, "EVALUATION", row, EVALUATION_FIELDS)


def main() -> None:
    load_dotenv_if_available()
    args = parse_args()
    num_questions = parse_num_questions(args.num_questions)
    tasks = select_tasks(args.level, args.question_type)
    answer_models = select_models(args.model)
    output_dir = output_dir_from_arg(args.output_dir)
    evaluation_output_path = output_dir / "all_evaluations.txt" if args.run_eval else None

    for task in tasks:
        validate_task(task)

    for task in tasks:
        for answer_model in answer_models:
            initialize_text_file(
                task_output_path(output_dir, task, answer_model),
                args.append,
                title=(
                    f"Results for level {task.level} / {task.question_type} / "
                    f"{answer_model.model_name}"
                ),
            )
    if evaluation_output_path is not None:
        initialize_text_file(
            evaluation_output_path,
            args.append,
            title=f"All evaluations using {OPENAI_EVAL_MODEL}",
        )

    print(f"Writing results to {relative_path(output_dir)}")
    for task in tasks:
        for answer_model in answer_models:
            run_task(
                task=task,
                answer_model=answer_model,
                num_questions=num_questions,
                output_dir=output_dir,
                evaluation_output_path=evaluation_output_path,
                run_eval=args.run_eval,
            )

    print(f"Done. Detailed text results are in {relative_path(output_dir)}")
    if evaluation_output_path is not None:
        print(f"All evaluations: {relative_path(evaluation_output_path)}")


if __name__ == "__main__":
    main()
