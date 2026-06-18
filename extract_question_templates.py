"""Extract structured question templates from raw question text files.

The script reads ``raw_questions/level_1``, ``raw_questions/level_2``, and
``raw_questions/level_3`` and writes one CSV per level under
``question_template/``.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_INPUT_DIR = Path("raw_questions")
DEFAULT_OUTPUT_DIR = Path("question_template")
LEVEL_FILES = ["level_1", "level_2", "level_3"]

QUESTION_RE = re.compile(r"^(?P<question>.*?)\s*\((?P<attributes>[^()]*)\)\s*$")


def quote_csv_field(value: str) -> str:
    """Quote one CSV field and escape embedded double quotes."""
    return f'"{value.replace(chr(34), chr(34) + chr(34))}"'


def normalize_question_text(question: str) -> str:
    """Normalize quote characters while preserving possessive wording."""
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
    }
    for old, new in replacements.items():
        question = question.replace(old, new)
    return re.sub(r"\s+", " ", question).strip()


def parse_section_tables(header: str) -> list[str]:
    """Parse a section header into table names."""
    return [part.strip() for part in header.split("+") if part.strip()]


def parse_attributes(attribute_text: str) -> list[str]:
    """Parse comma/semicolon separated attributes from the final parentheses."""
    attributes: list[str] = []
    for group in attribute_text.split(";"):
        for attribute in group.split(","):
            attribute = attribute.strip()
            if attribute:
                attributes.append(attribute)
    return attributes


def unique_in_order(values: list[str]) -> list[str]:
    """Return values in first-seen order without duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def row_signature(row: dict[str, object]) -> tuple[str, str, str, str]:
    """Return the stable fields used to match raw rows to existing templates."""
    return (
        str(row["question"]),
        str(row["num_tables"]),
        str(row["tables"]),
        str(row["attributes"]),
    )


def preserve_existing_question_ids(
    rows: list[dict[str, object]],
    existing_path: Path,
) -> list[dict[str, object]]:
    """Preserve existing template IDs when raw rows still match.

    This keeps intentional ID gaps, such as removed/deprecated templates, from
    being collapsed when templates are regenerated from raw question text.
    """
    if not existing_path.exists():
        return rows

    ids_by_signature: dict[tuple[str, str, str, str], list[int]] = {}
    with existing_path.open(newline="") as file:
        for existing_row in csv.DictReader(file):
            ids_by_signature.setdefault(row_signature(existing_row), []).append(
                int(existing_row["question_id"])
            )

    used_ids: set[int] = set()
    next_id = 0
    if ids_by_signature:
        next_id = max(question_id for ids in ids_by_signature.values() for question_id in ids) + 1

    preserved_rows: list[dict[str, object]] = []
    for row in rows:
        preserved = dict(row)
        candidates = ids_by_signature.get(row_signature(row), [])
        if candidates:
            question_id = candidates.pop(0)
        else:
            while next_id in used_ids:
                next_id += 1
            question_id = next_id
            next_id += 1
        preserved["question_id"] = question_id
        used_ids.add(question_id)
        preserved_rows.append(preserved)

    return preserved_rows


def infer_tables(attributes: list[str], section_tables: list[str]) -> list[str]:
    """Infer needed tables from qualified attributes, falling back to section tables."""
    qualified_tables = [
        attribute.split(".", 1)[0]
        for attribute in attributes
        if "." in attribute
    ]
    if qualified_tables:
        return unique_in_order(qualified_tables)
    return section_tables


def parse_question_block(
    block: list[str],
    section_tables: list[str],
    question_id: int,
) -> dict[str, object]:
    """Parse buffered question text plus trailing attribute parentheses."""
    text = " ".join(part.strip() for part in block if part.strip())
    match = QUESTION_RE.match(text)
    if not match:
        raise ValueError(f"Could not parse question block: {text!r}")

    question = normalize_question_text(match.group("question"))
    attributes = parse_attributes(match.group("attributes"))
    tables = infer_tables(attributes, section_tables)

    return {
        "question_id": question_id,
        "question": question,
        "num_tables": len(tables),
        "tables": "; ".join(tables),
        "attributes": "; ".join(attributes),
    }


def parse_level_file(path: Path) -> list[dict[str, object]]:
    """Parse one raw question level file."""
    rows: list[dict[str, object]] = []
    section_tables: list[str] = []
    block: list[str] = []

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("# "):
            if block:
                rows.append(parse_question_block(block, section_tables, len(rows)))
                block = []

            header = line[2:].strip()
            if header.lower().startswith("scope"):
                section_tables = []
                continue

            section_tables = parse_section_tables(header)
            continue

        if not section_tables and not block:
            raise ValueError(f"{path}:{line_number}: question found before a section header")

        block.append(line)
        if QUESTION_RE.match(" ".join(block)):
            rows.append(parse_question_block(block, section_tables, len(rows)))
            block = []

    if block:
        rows.append(parse_question_block(block, section_tables, len(rows)))

    return rows


def write_level_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write parsed rows with text/list columns quoted."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question_id", "question", "num_tables", "tables", "attributes"]
    with output_path.open("w", newline="") as file:
        file.write(",".join(fieldnames) + "\n")
        for row in rows:
            file.write(
                ",".join(
                    [
                        str(row["question_id"]),
                        quote_csv_field(str(row["question"])),
                        str(row["num_tables"]),
                        quote_csv_field(str(row["tables"])),
                        quote_csv_field(str(row["attributes"])),
                    ]
                )
                + "\n"
            )

def extract_templates(input_dir: Path, output_dir: Path) -> None:
    """Extract all configured levels."""
    for level_file in LEVEL_FILES:
        input_path = input_dir / level_file
        if not input_path.exists():
            raise FileNotFoundError(f"Missing raw question file: {input_path}")

        output_path = output_dir / f"{level_file}.csv"
        rows = preserve_existing_question_ids(parse_level_file(input_path), output_path)
        write_level_csv(rows, output_path)
        print(f"{level_file}: {len(rows):,} questions -> {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract question template CSVs from raw question files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing raw question files. Defaults to raw_questions/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV files will be written. Defaults to question_template/.",
    )
    return parser.parse_args()


def main() -> None:
    """Run extraction."""
    args = parse_args()
    extract_templates(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
