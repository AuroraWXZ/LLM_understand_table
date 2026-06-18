"""Materialize level 3 question answers with their supporting table rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_qa_experiment import (
    TableStore,
    filter_questions,
    find_truth_indices,
    load_json,
    truth_row_refs,
    write_json,
)


DEFAULT_QUESTIONS_PATH = Path("questions/level_3")
DEFAULT_DATASET_DIR = Path("dataset_clean")
DEFAULT_OUTPUT_DIR = Path("dataset_level3")
DEFAULT_OUTPUT_FILENAME = "level_3_answer_rows.json"
DEFAULT_SUMMARY_FILENAME = "summary.json"


def load_level3_questions(path: Path) -> list[dict[str, Any]]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"Question file must contain a JSON list: {path}")
    return records


def refs_by_table(record: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped_refs: dict[str, list[dict[str, Any]]] = {}
    for ref in truth_row_refs(record):
        table = str(ref.get("table", "")).strip()
        if not table:
            raise ValueError(f"Question {record.get('question_id')} has a row ref without a table")
        grouped_refs.setdefault(table, []).append(ref)
    return grouped_refs


def materialize_question(
    record: dict[str, Any],
    table_store: TableStore,
) -> dict[str, Any]:
    answer_tables = []
    for table_name, refs in refs_by_table(record).items():
        table = table_store.load(table_name)
        truth_indices = find_truth_indices(table, refs)
        rows = [
            {
                "row_index": index,
                "csv_row_number": index + 2,
                "values": {
                    field: table.rows[index].get(field, "")
                    for field in table.fieldnames
                },
            }
            for index in truth_indices
        ]
        answer_tables.append(
            {
                "table": table.name,
                "csv_file": table.csv_file,
                "csv_path": str(table.path),
                "columns": table.fieldnames,
                "row_refs": refs,
                "truth_row_indices": truth_indices,
                "truth_csv_row_numbers": [index + 2 for index in truth_indices],
                "rows": rows,
            }
        )

    return {
        "level": "level_3",
        "question_id": record.get("question_id"),
        "template_question_id": record.get("template_question_id"),
        "question": record.get("question"),
        "answer": record.get("answer"),
        "ground_truth": record.get("ground_truth"),
        "answer_tables": answer_tables,
    }


def build_dataset(
    records: list[dict[str, Any]],
    questions_path: Path,
    dataset_dir: Path,
) -> dict[str, Any]:
    table_store = TableStore(dataset_dir)
    questions = [
        materialize_question(record=record, table_store=table_store)
        for record in records
    ]
    return {
        "level": "level_3",
        "questions_path": str(questions_path),
        "dataset_dir": str(dataset_dir),
        "num_questions": len(questions),
        "questions": questions,
    }


def build_summary(dataset: dict[str, Any], output_path: Path) -> dict[str, Any]:
    table_counts: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    for question in dataset["questions"]:
        for table in question["answer_tables"]:
            table_name = table["table"]
            table_counts[table_name] = table_counts.get(table_name, 0) + 1
            row_counts[table_name] = row_counts.get(table_name, 0) + len(table["rows"])

    return {
        "level": dataset["level"],
        "num_questions": dataset["num_questions"],
        "output_path": str(output_path),
        "questions_path": dataset["questions_path"],
        "dataset_dir": dataset["dataset_dir"],
        "table_question_counts": dict(sorted(table_counts.items())),
        "table_row_counts": dict(sorted(row_counts.items())),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve questions/level_3 ground-truth row references to full CSV rows "
            "and store the materialized dataset in dataset_level3/."
        )
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to the level 3 question JSON file.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing the cleaned source CSV tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the materialized level 3 dataset should be written.",
    )
    parser.add_argument(
        "--output-filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help="Name of the JSON dataset file written inside --output-dir.",
    )
    parser.add_argument(
        "--summary-filename",
        default=DEFAULT_SUMMARY_FILENAME,
        help="Name of the JSON summary file written inside --output-dir.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=None,
        help="Only materialize this question id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Only materialize the first N selected questions.",
    )
    args = parser.parse_args(argv)
    if args.max_questions is not None and args.max_questions <= 0:
        parser.error("--max-questions must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    records = load_level3_questions(args.questions_path)
    question_ids = set(args.question_id) if args.question_id else None
    records = filter_questions(records, question_ids, args.max_questions)

    dataset = build_dataset(
        records=records,
        questions_path=args.questions_path,
        dataset_dir=args.dataset_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_filename
    summary_path = args.output_dir / args.summary_filename
    write_json(output_path, dataset)
    write_json(summary_path, build_summary(dataset, output_path))

    print(
        json.dumps(
            {
                "num_questions": dataset["num_questions"],
                "output_path": str(output_path),
                "summary_path": str(summary_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
