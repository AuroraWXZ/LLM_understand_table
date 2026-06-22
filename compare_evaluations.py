"""Compare original and counterfactual evaluation results.

The script matches per-question JSON files in ``evaluation/`` and
``counter_evaluation/`` for the same model, level, and question id. It reports
questions whose evaluator outcome changed between the two roots.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_EVALUATION_ROOT = Path("evaluation")
DEFAULT_COUNTER_EVALUATION_ROOT = Path("counter_evaluation")
DEFAULT_OUTPUT = Path("evaluation_counter_comparison_report.md")


@dataclass(frozen=True)
class ComparisonRecord:
    model: str
    level: str
    question_id: str
    evaluation_path: Path
    counter_evaluation_path: Path
    evaluation: dict[str, Any]
    counter_evaluation: dict[str, Any]


@dataclass(frozen=True)
class LevelComparison:
    model: str
    level: str
    matched_count: int
    evaluation_accuracy_sum: float
    evaluation_accuracy_count: int
    counter_accuracy_sum: float
    counter_accuracy_count: int
    changed_records: list[ComparisonRecord]
    missing_in_evaluation: list[str]
    missing_in_counter_evaluation: list[str]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_question_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def question_id_sort_key(question_id: str) -> tuple[int, int | str]:
    if re.fullmatch(r"-?\d+", question_id):
        return (0, int(question_id))
    return (1, question_id)


def level_sort_key(level: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"level_(\d+)", level)
    if match:
        return (0, int(match.group(1)))
    return (1, level)


def discover_models(evaluation_root: Path, counter_root: Path) -> list[str]:
    evaluation_models = {
        path.name for path in evaluation_root.iterdir() if path.is_dir()
    } if evaluation_root.exists() else set()
    counter_models = {
        path.name for path in counter_root.iterdir() if path.is_dir()
    } if counter_root.exists() else set()
    return sorted(evaluation_models & counter_models)


def discover_levels(evaluation_root: Path, counter_root: Path, model: str) -> list[str]:
    evaluation_model_root = evaluation_root / model
    counter_model_root = counter_root / model
    evaluation_levels = {
        path.name
        for path in evaluation_model_root.iterdir()
        if path.is_dir() and path.name.startswith("level_")
    } if evaluation_model_root.exists() else set()
    counter_levels = {
        path.name
        for path in counter_model_root.iterdir()
        if path.is_dir() and path.name.startswith("level_")
    } if counter_model_root.exists() else set()
    return sorted(evaluation_levels & counter_levels, key=level_sort_key)


def load_question_docs(directory: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    docs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        question_id = normalize_question_id(payload.get("question_id"))
        if not question_id:
            question_id = path.stem
        docs[question_id] = (path, payload)
    return docs


def normalize_correct(value: Any) -> bool | str | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "correct"}:
        return True
    if text in {"false", "no", "0", "incorrect"}:
        return False
    return str(value)


def normalize_accuracy(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return str(value)


def numeric_accuracy(payload: dict[str, Any]) -> float | None:
    accuracy = normalize_accuracy(payload.get("accuracy"))
    if isinstance(accuracy, (int, float)):
        return float(accuracy)
    return None


def average_accuracy(accuracy_sum: float, accuracy_count: int) -> float | None:
    if accuracy_count <= 0:
        return None
    return accuracy_sum / accuracy_count


def format_accuracy(accuracy_sum: float, accuracy_count: int) -> str:
    average = average_accuracy(accuracy_sum, accuracy_count)
    if average is None:
        return "N/A"
    return f"{average:.4f}"


def format_accuracy_count(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}"


def format_accuracy_with_count(accuracy_sum: float, accuracy_count: int) -> str:
    average = average_accuracy(accuracy_sum, accuracy_count)
    if average is None:
        return "N/A"
    return f"{average:.4f} ({format_accuracy_count(accuracy_sum)}/{accuracy_count})"


def format_accuracy_delta(
    evaluation_sum: float,
    evaluation_count: int,
    counter_sum: float,
    counter_count: int,
) -> str:
    evaluation_average = average_accuracy(evaluation_sum, evaluation_count)
    counter_average = average_accuracy(counter_sum, counter_count)
    if evaluation_average is None or counter_average is None:
        return "N/A"
    return f"{counter_average - evaluation_average:+.4f}"


def evaluation_result_changed(
    evaluation: dict[str, Any],
    counter_evaluation: dict[str, Any],
) -> bool:
    evaluation_correct = normalize_correct(evaluation.get("correct"))
    counter_correct = normalize_correct(counter_evaluation.get("correct"))
    evaluation_accuracy = normalize_accuracy(evaluation.get("accuracy"))
    counter_accuracy = normalize_accuracy(counter_evaluation.get("accuracy"))
    return (
        evaluation_correct != counter_correct
        or evaluation_accuracy != counter_accuracy
    )


def compare_level(
    evaluation_root: Path,
    counter_root: Path,
    model: str,
    level: str,
) -> LevelComparison:
    evaluation_dir = evaluation_root / model / level
    counter_dir = counter_root / model / level
    evaluation_docs = load_question_docs(evaluation_dir)
    counter_docs = load_question_docs(counter_dir)

    evaluation_ids = set(evaluation_docs)
    counter_ids = set(counter_docs)
    matched_ids = sorted(evaluation_ids & counter_ids, key=question_id_sort_key)
    changed_records: list[ComparisonRecord] = []
    evaluation_accuracy_sum = 0.0
    evaluation_accuracy_count = 0
    counter_accuracy_sum = 0.0
    counter_accuracy_count = 0

    for question_id in matched_ids:
        evaluation_path, evaluation = evaluation_docs[question_id]
        counter_path, counter_evaluation = counter_docs[question_id]
        evaluation_accuracy = numeric_accuracy(evaluation)
        if evaluation_accuracy is not None:
            evaluation_accuracy_sum += evaluation_accuracy
            evaluation_accuracy_count += 1
        counter_accuracy = numeric_accuracy(counter_evaluation)
        if counter_accuracy is not None:
            counter_accuracy_sum += counter_accuracy
            counter_accuracy_count += 1
        if evaluation_result_changed(evaluation, counter_evaluation):
            changed_records.append(
                ComparisonRecord(
                    model=model,
                    level=level,
                    question_id=question_id,
                    evaluation_path=evaluation_path,
                    counter_evaluation_path=counter_path,
                    evaluation=evaluation,
                    counter_evaluation=counter_evaluation,
                )
            )

    return LevelComparison(
        model=model,
        level=level,
        matched_count=len(matched_ids),
        evaluation_accuracy_sum=evaluation_accuracy_sum,
        evaluation_accuracy_count=evaluation_accuracy_count,
        counter_accuracy_sum=counter_accuracy_sum,
        counter_accuracy_count=counter_accuracy_count,
        changed_records=changed_records,
        missing_in_evaluation=sorted(counter_ids - evaluation_ids, key=question_id_sort_key),
        missing_in_counter_evaluation=sorted(evaluation_ids - counter_ids, key=question_id_sort_key),
    )


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def md_escape(value: Any) -> str:
    return as_text(value).replace("|", r"\|").replace("\n", "<br>")


def quote_block(value: Any) -> str:
    text = as_text(value).strip()
    if not text:
        return "> "
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def result_label(payload: dict[str, Any]) -> str:
    correct = normalize_correct(payload.get("correct"))
    accuracy = normalize_accuracy(payload.get("accuracy"))
    return f"correct={correct}, accuracy={accuracy}"


def transition_key(record: ComparisonRecord) -> str:
    before = normalize_correct(record.evaluation.get("correct"))
    after = normalize_correct(record.counter_evaluation.get("correct"))
    if before is True and after is False:
        return "correct_to_incorrect"
    if before is False and after is True:
        return "incorrect_to_correct"
    return "other_change"


def summarize_comparisons(comparisons: list[LevelComparison]) -> dict[str, dict[str, int | float]]:
    summary: dict[str, dict[str, int | float]] = {}
    for comparison in comparisons:
        model_summary = summary.setdefault(
            comparison.model,
            {
                "matched": 0,
                "evaluation_accuracy_sum": 0.0,
                "evaluation_accuracy_count": 0,
                "counter_accuracy_sum": 0.0,
                "counter_accuracy_count": 0,
                "changed": 0,
                "correct_to_incorrect": 0,
                "incorrect_to_correct": 0,
                "other_change": 0,
                "missing_in_evaluation": 0,
                "missing_in_counter_evaluation": 0,
            },
        )
        model_summary["matched"] += comparison.matched_count
        model_summary["evaluation_accuracy_sum"] += comparison.evaluation_accuracy_sum
        model_summary["evaluation_accuracy_count"] += comparison.evaluation_accuracy_count
        model_summary["counter_accuracy_sum"] += comparison.counter_accuracy_sum
        model_summary["counter_accuracy_count"] += comparison.counter_accuracy_count
        model_summary["changed"] += len(comparison.changed_records)
        model_summary["missing_in_evaluation"] += len(comparison.missing_in_evaluation)
        model_summary["missing_in_counter_evaluation"] += len(
            comparison.missing_in_counter_evaluation
        )
        for record in comparison.changed_records:
            model_summary[transition_key(record)] += 1
    return summary


def render_report(
    comparisons: list[LevelComparison],
    evaluation_root: Path,
    counter_root: Path,
    models: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Evaluation vs Counter Evaluation Comparison")
    lines.append("")
    lines.append(f"- Evaluation root: `{evaluation_root}`")
    lines.append(f"- Counter evaluation root: `{counter_root}`")
    lines.append(f"- Models: {', '.join(f'`{model}`' for model in models)}")
    lines.append("")

    totals = summarize_comparisons(comparisons)
    lines.append("## Summary by model")
    lines.append("")
    lines.append("Accuracy is shown as average (sum/evaluated). Model rows are weighted across all compared levels.")
    lines.append("")
    lines.append(
        "| Model | Matched questions | Original accuracy (sum/n) | Counter accuracy (sum/n) | Accuracy delta | Changed results | Original correct -> counter incorrect | Original incorrect -> counter correct | Other changes | Missing only in counter | Missing only in original |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in models:
        model_summary = totals.get(model, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(model),
                    str(model_summary.get("matched", 0)),
                    format_accuracy_with_count(
                        float(model_summary.get("evaluation_accuracy_sum", 0.0)),
                        int(model_summary.get("evaluation_accuracy_count", 0)),
                    ),
                    format_accuracy_with_count(
                        float(model_summary.get("counter_accuracy_sum", 0.0)),
                        int(model_summary.get("counter_accuracy_count", 0)),
                    ),
                    format_accuracy_delta(
                        float(model_summary.get("evaluation_accuracy_sum", 0.0)),
                        int(model_summary.get("evaluation_accuracy_count", 0)),
                        float(model_summary.get("counter_accuracy_sum", 0.0)),
                        int(model_summary.get("counter_accuracy_count", 0)),
                    ),
                    str(model_summary.get("changed", 0)),
                    str(model_summary.get("correct_to_incorrect", 0)),
                    str(model_summary.get("incorrect_to_correct", 0)),
                    str(model_summary.get("other_change", 0)),
                    str(model_summary.get("missing_in_evaluation", 0)),
                    str(model_summary.get("missing_in_counter_evaluation", 0)),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Summary by level")
    lines.append("")
    lines.append(
        "| Model | Level | Matched questions | Original accuracy (sum/n) | Counter accuracy (sum/n) | Accuracy delta | Changed results | Original correct -> counter incorrect | Original incorrect -> counter correct | Other changes |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for comparison in comparisons:
        transition_counts = {
            "correct_to_incorrect": 0,
            "incorrect_to_correct": 0,
            "other_change": 0,
        }
        for record in comparison.changed_records:
            transition_counts[transition_key(record)] += 1
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(comparison.model),
                    md_escape(comparison.level),
                    str(comparison.matched_count),
                    format_accuracy_with_count(
                        comparison.evaluation_accuracy_sum,
                        comparison.evaluation_accuracy_count,
                    ),
                    format_accuracy_with_count(
                        comparison.counter_accuracy_sum,
                        comparison.counter_accuracy_count,
                    ),
                    format_accuracy_delta(
                        comparison.evaluation_accuracy_sum,
                        comparison.evaluation_accuracy_count,
                        comparison.counter_accuracy_sum,
                        comparison.counter_accuracy_count,
                    ),
                    str(len(comparison.changed_records)),
                    str(transition_counts["correct_to_incorrect"]),
                    str(transition_counts["incorrect_to_correct"]),
                    str(transition_counts["other_change"]),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Changed question IDs by level")
    lines.append("")
    lines.append("| Model | Level | Question IDs |")
    lines.append("| --- | --- | --- |")
    for comparison in comparisons:
        changed_ids = ", ".join(record.question_id for record in comparison.changed_records)
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(comparison.model),
                    md_escape(comparison.level),
                    md_escape(changed_ids or "None"),
                ]
            )
            + " |"
        )
    lines.append("")

    missing_comparisons = [
        comparison
        for comparison in comparisons
        if comparison.missing_in_evaluation or comparison.missing_in_counter_evaluation
    ]
    if missing_comparisons:
        lines.append("## Missing files")
        lines.append("")
        for comparison in missing_comparisons:
            lines.append(f"### {comparison.model} / {comparison.level}")
            if comparison.missing_in_evaluation:
                missing = ", ".join(comparison.missing_in_evaluation)
                lines.append(f"- Present only in counter evaluation: {missing}")
            if comparison.missing_in_counter_evaluation:
                missing = ", ".join(comparison.missing_in_counter_evaluation)
                lines.append(f"- Present only in original evaluation: {missing}")
            lines.append("")

    lines.append("## Changed questions")
    lines.append("")
    changed_total = sum(len(comparison.changed_records) for comparison in comparisons)
    if not changed_total:
        lines.append("No questions changed evaluator result between the two evaluation roots.")
        lines.append("")
        return "\n".join(lines)

    for comparison in comparisons:
        if not comparison.changed_records:
            continue
        lines.append(
            f"### {comparison.model} / {comparison.level} ({len(comparison.changed_records)} changed)"
        )
        lines.append("")
        for record in comparison.changed_records:
            evaluation = record.evaluation
            counter = record.counter_evaluation
            question = evaluation.get("question") or counter.get("question") or ""
            lines.append(f"#### Question {record.question_id}")
            lines.append("")
            lines.append(f"- Question: {as_text(question)}")
            lines.append(f"- Original evaluation: `{result_label(evaluation)}`")
            lines.append(f"- Counter evaluation: `{result_label(counter)}`")
            lines.append(f"- Original file: `{record.evaluation_path}`")
            lines.append(f"- Counter file: `{record.counter_evaluation_path}`")
            lines.append("")
            lines.append("Original reference answer:")
            lines.append("")
            lines.append(quote_block(evaluation.get("reference_answer")))
            lines.append("")
            lines.append("Original reference explanation:")
            lines.append("")
            lines.append(quote_block(evaluation.get("reference_explanation")))
            lines.append("")
            lines.append("Original model answer:")
            lines.append("")
            lines.append(quote_block(evaluation.get("model_answer")))
            lines.append("")
            lines.append("Original evaluator rationale:")
            lines.append("")
            lines.append(quote_block(evaluation.get("rationale")))
            lines.append("")
            lines.append("Counter reference answer:")
            lines.append("")
            lines.append(quote_block(counter.get("reference_answer")))
            lines.append("")
            lines.append("Counter reference explanation:")
            lines.append("")
            lines.append(quote_block(counter.get("reference_explanation")))
            lines.append("")
            lines.append("Counter model answer:")
            lines.append("")
            lines.append(quote_block(counter.get("model_answer")))
            lines.append("")
            lines.append("Counter evaluator rationale:")
            lines.append("")
            lines.append(quote_block(counter.get("rationale")))
            lines.append("")

    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-question JSON documents in evaluation/ and "
            "counter_evaluation/ and report changed evaluator outcomes."
        )
    )
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument(
        "--counter-evaluation-root",
        type=Path,
        default=DEFAULT_COUNTER_EVALUATION_ROOT,
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model directory to compare. Can be passed multiple times. "
            "Defaults to model directories present in both roots."
        ),
    )
    parser.add_argument(
        "--level",
        action="append",
        default=None,
        help=(
            "Question level to compare. Can be passed multiple times. "
            "Defaults to levels present for each selected model in both roots."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    evaluation_root = args.evaluation_root
    counter_root = args.counter_evaluation_root

    if not evaluation_root.exists():
        print(f"Evaluation root does not exist: {evaluation_root}", file=sys.stderr)
        return 1
    if not counter_root.exists():
        print(f"Counter evaluation root does not exist: {counter_root}", file=sys.stderr)
        return 1

    models = args.model or discover_models(evaluation_root, counter_root)
    if not models:
        print("No shared model directories found to compare.", file=sys.stderr)
        return 1

    comparisons: list[LevelComparison] = []
    for model in models:
        levels = args.level or discover_levels(evaluation_root, counter_root, model)
        if not levels:
            print(f"No shared levels found for model: {model}", file=sys.stderr)
            continue
        for level in levels:
            comparisons.append(compare_level(evaluation_root, counter_root, model, level))

    if not comparisons:
        print("No model/level pairs were compared.", file=sys.stderr)
        return 1

    report = render_report(
        comparisons=comparisons,
        evaluation_root=evaluation_root,
        counter_root=counter_root,
        models=models,
    )
    write_text(args.output, report)

    changed_total = sum(len(comparison.changed_records) for comparison in comparisons)
    matched_total = sum(comparison.matched_count for comparison in comparisons)
    print(
        f"Wrote {args.output} with {changed_total} changed results "
        f"across {matched_total} matched question files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
