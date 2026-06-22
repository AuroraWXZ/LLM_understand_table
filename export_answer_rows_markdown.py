"""Export answer-supporting rows for generated questions to Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_qa_experiment import TableStore, rows_to_csv, truth_row_refs


DEFAULT_LEVELS = ["level_1", "level_2", "level_3"]
DEFAULT_ORIGINAL_QUESTIONS_ROOT = Path("questions")
DEFAULT_COUNTER_QUESTIONS_ROOT = Path("question_counter")
DEFAULT_ORIGINAL_DATASET_DIR = Path("dataset_clean")
DEFAULT_COUNTER_DATASET_DIR = Path("dataset_counter")
DEFAULT_OUTPUT_PATH = Path("dataset_answer_rows.md")
DEFAULT_OUTPUT_DIR = Path(".")
DEFAULT_OUTPUT_TEMPLATE = "dataset_{level_slug}_answer_rows_{variant}.md"


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


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
    raise FileNotFoundError(f"Could not find questions for {level}; tried {tried}")


def refs_by_table(question: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ref in truth_row_refs(question):
        table = str(ref.get("table", "")).strip()
        if table:
            grouped.setdefault(table, []).append(ref)
    return grouped


def ref_identity_parts(ref: dict[str, Any]) -> tuple[list[str], list[str]]:
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


class TruthIndexResolver:
    def __init__(self) -> None:
        self._lookup_cache: dict[
            tuple[str, tuple[str, ...]],
            dict[tuple[str, ...], list[int]],
        ] = {}

    def find(self, table: Any, refs: list[dict[str, Any]]) -> list[int]:
        indices: list[int] = []
        seen: set[int] = set()

        for ref in refs:
            attributes, values = ref_identity_parts(ref)
            attribute_key = tuple(attributes)
            value_key = tuple(values)
            cache_key = (str(table.path), attribute_key)

            if cache_key not in self._lookup_cache:
                missing = [
                    attribute
                    for attribute in attributes
                    if attribute not in table.fieldnames
                ]
                if missing:
                    raise KeyError(
                        f"{table.name}: id attribute(s) {missing!r} are not in "
                        f"{table.csv_file}"
                    )

                lookup: dict[tuple[str, ...], list[int]] = {}
                for index, row in enumerate(table.rows):
                    row_key = tuple(
                        str(row.get(attribute, "")) for attribute in attributes
                    )
                    lookup.setdefault(row_key, []).append(index)
                self._lookup_cache[cache_key] = lookup

            matches = self._lookup_cache[cache_key].get(value_key, [])
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


def materialize_answer_tables(
    question: dict[str, Any],
    table_store: TableStore,
    truth_resolver: TruthIndexResolver,
) -> list[dict[str, Any]]:
    answer_tables: list[dict[str, Any]] = []
    for table_name, refs in refs_by_table(question).items():
        table = table_store.load(table_name)
        truth_indices = truth_resolver.find(table, refs)
        answer_tables.append(
            {
                "table": table.name,
                "csv_file": table.csv_file,
                "csv_path": str(table.path),
                "columns": table.fieldnames,
                "truth_csv_row_numbers": [index + 2 for index in truth_indices],
                "rows": [
                    {
                        "csv_row_number": index + 2,
                        "values": table.rows[index],
                    }
                    for index in truth_indices
                ],
            }
        )
    return answer_tables


def display_level(level: str) -> str:
    return level.replace("_", " ").title()


def output_level_slug(level: str) -> str:
    return level.replace("_", "")


def resolve_output_path(
    output_dir: Path,
    output_template: str,
    level: str,
    variant: str,
) -> Path:
    return output_dir / output_template.format(
        level=level,
        level_slug=output_level_slug(level),
        variant=variant,
    )


def answer_table_to_markdown(
    table: dict[str, Any],
    heading: str = "####",
) -> list[str]:
    table_name = table.get("table", "")
    csv_file = table.get("csv_file", "")
    csv_path = table.get("csv_path", "")
    columns = [str(column) for column in table.get("columns", [])]
    rows = [
        row.get("values", {})
        for row in table.get("rows", [])
        if isinstance(row, dict) and isinstance(row.get("values", {}), dict)
    ]
    csv_row_numbers = table.get("truth_csv_row_numbers", [])

    return [
        f"{heading} Table: {table_name} ({csv_file})",
        "",
        f"- CSV path: `{csv_path}`",
        f"- Source CSV row numbers: {', '.join(str(value) for value in csv_row_numbers) or 'none'}",
        "",
        "````csv",
        rows_to_csv(columns, rows) if columns else "",
        "````",
        "",
    ]


def questions_to_markdown(
    questions_path: Path,
    dataset_dir: Path,
    question_heading: str,
    table_heading: str,
    table_store: TableStore | None = None,
    truth_resolver: TruthIndexResolver | None = None,
) -> list[str]:
    questions = load_questions(questions_path)
    table_store = table_store or TableStore(dataset_dir)
    truth_resolver = truth_resolver or TruthIndexResolver()
    lines = [
        f"- Questions path: `{questions_path}`",
        f"- Dataset directory: `{dataset_dir}`",
        f"- Number of questions: {len(questions)}",
        "",
    ]

    for question in questions:
        question_id = question.get("question_id", "")
        template_id = question.get("template_question_id", "")
        lines.extend(
            [
                f"{question_heading} Question {question_id} / Template {template_id}",
                "",
                f"**Question:** {question.get('question', '')}",
                "",
                f"**Answer:** {question.get('answer', '')}",
                "",
                f"**Explanation:** {question.get('explanation', '')}",
                "",
            ]
        )
        for table in materialize_answer_tables(question, table_store, truth_resolver):
            lines.extend(answer_table_to_markdown(table, heading=table_heading))

    return lines


def level_to_markdown(
    title: str,
    questions_path: Path,
    dataset_dir: Path,
    level: str,
    table_store: TableStore | None = None,
    truth_resolver: TruthIndexResolver | None = None,
) -> list[str]:
    lines = [
        f"### {title} / {level}",
        "",
    ]
    lines.extend(
        questions_to_markdown(
            questions_path,
            dataset_dir,
            question_heading="####",
            table_heading="####",
            table_store=table_store,
            truth_resolver=truth_resolver,
        )
    )
    return lines


def single_level_to_markdown(
    variant_label: str,
    questions_path: Path,
    dataset_dir: Path,
    level: str,
    table_store: TableStore | None = None,
    truth_resolver: TruthIndexResolver | None = None,
) -> list[str]:
    lines = [
        f"# {display_level(level)} Answer Rows - {variant_label}",
        "",
    ]
    lines.extend(
        questions_to_markdown(
            questions_path,
            dataset_dir,
            question_heading="###",
            table_heading="####",
            table_store=table_store,
            truth_resolver=truth_resolver,
        )
    )
    return lines


def dataset_to_markdown(
    title: str,
    questions_root: Path,
    dataset_dir: Path,
    levels: list[str],
) -> list[str]:
    table_store = TableStore(dataset_dir)
    truth_resolver = TruthIndexResolver()
    lines = [
        f"## {title}",
        "",
        f"- Questions root: `{questions_root}`",
        f"- Dataset directory: `{dataset_dir}`",
        f"- Levels: {', '.join(levels)}",
        "",
    ]
    for level in levels:
        questions_path = resolve_questions_path(questions_root, level)
        lines.extend(
            level_to_markdown(
                title,
                questions_path,
                dataset_dir,
                level,
                table_store=table_store,
                truth_resolver=truth_resolver,
            )
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write original and counterfactual answer-supporting rows for "
            "generated questions to Markdown files."
        )
    )
    parser.add_argument(
        "--level",
        action="append",
        choices=DEFAULT_LEVELS,
        default=None,
        help="Question level to export. Can be passed multiple times. Defaults to all levels.",
    )
    parser.add_argument(
        "--original-questions-root",
        type=Path,
        default=DEFAULT_ORIGINAL_QUESTIONS_ROOT,
    )
    parser.add_argument(
        "--counter-questions-root",
        type=Path,
        default=DEFAULT_COUNTER_QUESTIONS_ROOT,
    )
    parser.add_argument(
        "--original-dataset-dir",
        type=Path,
        default=DEFAULT_ORIGINAL_DATASET_DIR,
    )
    parser.add_argument(
        "--counter-dataset-dir",
        type=Path,
        default=DEFAULT_COUNTER_DATASET_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Combined Markdown file to write.",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not write the combined Markdown file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-level Markdown files.",
    )
    parser.add_argument(
        "--output-template",
        default=DEFAULT_OUTPUT_TEMPLATE,
        help=(
            "Filename template for per-level files. Available fields: "
            "{level}, {level_slug}, {variant}."
        ),
    )
    return parser.parse_args()


def write_combined_markdown(args: argparse.Namespace, levels: list[str]) -> Path:
    lines = [
        "# Answer Rows",
        "",
        "This file materializes the answer-supporting rows for original and counterfactual questions.",
        "",
    ]
    lines.extend(
        dataset_to_markdown(
            "Original",
            args.original_questions_root,
            args.original_dataset_dir,
            levels,
        )
    )
    lines.extend(
        dataset_to_markdown(
            "Counterfactual",
            args.counter_questions_root,
            args.counter_dataset_dir,
            levels,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return args.output


def write_split_markdowns(args: argparse.Namespace, levels: list[str]) -> list[Path]:
    variants = [
        (
            "original",
            "Original",
            args.original_questions_root,
            args.original_dataset_dir,
        ),
        (
            "counter",
            "Counter",
            args.counter_questions_root,
            args.counter_dataset_dir,
        ),
    ]
    table_stores: dict[tuple[str, str], TableStore] = {}
    truth_resolvers: dict[tuple[str, str], TruthIndexResolver] = {}
    written: list[Path] = []
    for level in levels:
        for variant, variant_label, questions_root, dataset_dir in variants:
            cache_key = (variant, str(dataset_dir))
            table_store = table_stores.setdefault(cache_key, TableStore(dataset_dir))
            truth_resolver = truth_resolvers.setdefault(
                cache_key,
                TruthIndexResolver(),
            )
            questions_path = resolve_questions_path(questions_root, level)
            output_path = resolve_output_path(
                args.output_dir,
                args.output_template,
                level,
                variant,
            )
            lines = single_level_to_markdown(
                variant_label,
                questions_path,
                dataset_dir,
                level,
                table_store=table_store,
                truth_resolver=truth_resolver,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "\n".join(lines).rstrip() + "\n",
                encoding="utf-8",
            )
            written.append(output_path)
    return written


def main() -> None:
    args = parse_args()
    levels = args.level or DEFAULT_LEVELS

    for output_path in write_split_markdowns(args, levels):
        print(f"Wrote {output_path}")
    if not args.no_combined:
        print(f"Wrote {write_combined_markdown(args, levels)}")


if __name__ == "__main__":
    main()
