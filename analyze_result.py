"""Analyze model accuracy by benchmark level from text result files."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_RESULTS_PATH = Path("results")
RECORD_HEADER_RE = re.compile(r"^==========\s+([A-Z_]+)\s+==========$")
FIELD_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(.*)$")

KNOWN_FIELDS = {
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
}


@dataclass
class AccuracyStats:
    records: int = 0
    correct: int = 0
    incorrect: int = 0
    missing_eval: int = 0

    @property
    def evaluated(self) -> int:
        return self.correct + self.incorrect

    def accuracy(self, missing_as_incorrect: bool) -> float | None:
        denominator = self.records if missing_as_incorrect else self.evaluated
        if denominator == 0:
            return None
        return self.correct / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Output the accuracy of each model within each benchmark level."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=(
            "Result file or result directory to analyze. Defaults to results/. "
            "When a directory is provided, level_*.txt files are used first."
        ),
    )
    parser.add_argument(
        "--by-question-type",
        action="store_true",
        help="Break results down by level, question type, and model.",
    )
    parser.add_argument(
        "--missing-as-incorrect",
        action="store_true",
        help="Count blank or unparsable eval_correct values as incorrect in accuracy.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        help="Optional path to save the summary table as CSV.",
    )
    return parser.parse_args()


def resolve_result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Result path does not exist: {path}")

    level_files = sorted(path.glob("level_*.txt"))
    if level_files:
        return level_files

    combined = path / "all_evaluations.txt"
    if combined.exists():
        return [combined]

    files = sorted(path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt result files found in: {path}")
    return files


def parse_text_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    current_field: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        header_match = RECORD_HEADER_RE.match(raw_line)
        if header_match:
            if current is not None:
                records.append(current)
            current = {"record_type": header_match.group(1)}
            current_field = None
            continue

        if current is None:
            continue

        field_match = FIELD_RE.match(raw_line)
        if field_match and field_match.group(1) in KNOWN_FIELDS:
            current_field = field_match.group(1)
            current[current_field] = field_match.group(2).lstrip()
            continue

        if current_field is not None:
            previous = current.get(current_field, "")
            current[current_field] = f"{previous}\n{raw_line}" if previous else raw_line

    if current is not None:
        records.append(current)
    return records


def parse_eval_correct(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def level_sort_key(level: str) -> tuple[int, str]:
    try:
        return (int(level), level)
    except ValueError:
        return (10_000, level)


def collect_stats(
    records: Iterable[dict[str, str]],
    by_question_type: bool,
) -> dict[tuple[str, ...], AccuracyStats]:
    grouped: dict[tuple[str, ...], AccuracyStats] = defaultdict(AccuracyStats)

    for record in records:
        level = record.get("level", "").strip()
        answer_model = record.get("answer_model", "").strip()
        if not level or not answer_model:
            continue

        if by_question_type:
            key = (
                level,
                record.get("question_type", "").strip() or "(unknown)",
                answer_model,
            )
        else:
            key = (level, answer_model)

        stats = grouped[key]
        stats.records += 1

        eval_correct = parse_eval_correct(record.get("eval_correct", ""))
        if eval_correct is True:
            stats.correct += 1
        elif eval_correct is False:
            stats.incorrect += 1
        else:
            stats.missing_eval += 1

    return dict(grouped)


def summary_rows(
    grouped: dict[tuple[str, ...], AccuracyStats],
    by_question_type: bool,
    missing_as_incorrect: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, stats in sorted(
        grouped.items(),
        key=lambda item: (
            level_sort_key(item[0][0]),
            item[0][1:-1],
            item[0][-1],
        ),
    ):
        accuracy = stats.accuracy(missing_as_incorrect)
        row = {
            "level": key[0],
            "answer_model": key[-1],
            "records": str(stats.records),
            "evaluated": str(stats.evaluated),
            "correct": str(stats.correct),
            "incorrect": str(stats.incorrect),
            "missing_eval": str(stats.missing_eval),
            "accuracy": "N/A" if accuracy is None else f"{accuracy:.2%}",
        }
        if by_question_type:
            row["question_type"] = key[1]
        rows.append(row)
    return rows


def print_table(rows: list[dict[str, str]], by_question_type: bool) -> None:
    if not rows:
        print("No analyzable result records found.")
        return

    columns = ["level"]
    if by_question_type:
        columns.append("question_type")
    columns.extend(
        [
            "answer_model",
            "records",
            "evaluated",
            "correct",
            "incorrect",
            "missing_eval",
            "accuracy",
        ]
    )

    widths = {
        column: max(len(column), *(len(row.get(column, "")) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    separator = "  ".join("-" * widths[column] for column in columns)

    print(header)
    print(separator)
    for row in rows:
        print("  ".join(row.get(column, "").ljust(widths[column]) for column in columns))


def write_csv(path: Path, rows: list[dict[str, str]], by_question_type: bool) -> None:
    columns = ["level"]
    if by_question_type:
        columns.append("question_type")
    columns.extend(
        [
            "answer_model",
            "records",
            "evaluated",
            "correct",
            "incorrect",
            "missing_eval",
            "accuracy",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    result_files = resolve_result_files(args.path)
    records: list[dict[str, str]] = []
    for path in result_files:
        records.extend(parse_text_records(path))

    grouped = collect_stats(records, by_question_type=args.by_question_type)
    rows = summary_rows(
        grouped,
        by_question_type=args.by_question_type,
        missing_as_incorrect=args.missing_as_incorrect,
    )

    print_table(rows, by_question_type=args.by_question_type)
    if args.csv_path:
        write_csv(args.csv_path, rows, by_question_type=args.by_question_type)
        print(f"\nWrote CSV summary to {args.csv_path}")


if __name__ == "__main__":
    main()
