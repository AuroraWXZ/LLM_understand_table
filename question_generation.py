"""Generate benchmark questions from templates and seed files."""

from __future__ import annotations

import argparse
from pathlib import Path

from question_template.question_generation.level_1 import generate_and_write_level_1
from question_template.question_generation.level_2 import generate_and_write_level_2
from question_template.question_generation.level_3 import generate_and_write_level_3


DEFAULT_TEMPLATE_DIR = Path("question_template")
DEFAULT_SEED_DIR = Path("question_template/seed")
ORIGINAL_DATASET_DIR = Path("dataset_clean")
ORIGINAL_OUTPUT_DIR = Path("questions")
COUNTER_DATASET_DIR = Path("dataset_counter")
COUNTER_OUTPUT_DIR = Path("question_counter")
DEFAULT_LEVELS = ["level_1"]


def resolve_dataset_paths(
    variant: str,
    dataset_dir: Path | None,
    output_dir: Path | None,
) -> tuple[Path, Path]:
    """Resolve dataset and output directories for original or counter questions."""
    if variant == "counter":
        default_dataset_dir = COUNTER_DATASET_DIR
        default_output_dir = COUNTER_OUTPUT_DIR
    elif variant == "original":
        default_dataset_dir = ORIGINAL_DATASET_DIR
        default_output_dir = ORIGINAL_OUTPUT_DIR
    else:
        raise ValueError(f"Unsupported dataset variant: {variant}")

    return dataset_dir or default_dataset_dir, output_dir or default_output_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark questions with answers and provenance."
    )
    parser.add_argument(
        "--level",
        action="append",
        choices=["level_1", "level_2", "level_3"],
        help="Level to generate. Can be passed multiple times. Defaults to level_1.",
    )
    parser.add_argument(
        "--variant",
        "--dataset",
        choices=["counter", "original"],
        default="original",
        help=(
            "Question/data variant to use. 'counter' uses question_counter/ "
            "and dataset_counter/; 'original' uses questions/ and dataset_clean/. "
            "Defaults to original."
        ),
    )
    parser.add_argument(
        "--counter",
        action="store_const",
        const="counter",
        dest="variant",
        help="Shortcut for --variant counter.",
    )
    parser.add_argument(
        "--original",
        action="store_const",
        const="original",
        dest="variant",
        help="Shortcut for --variant original.",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=DEFAULT_TEMPLATE_DIR,
        help="Directory containing question template CSVs.",
    )
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=DEFAULT_SEED_DIR,
        help="Directory containing question seed txt files.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Override the dataset CSV directory selected by --variant.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the question output directory selected by --variant.",
    )
    return parser.parse_args()


def main() -> None:
    """Run question generation."""
    args = parse_args()
    levels = args.level or DEFAULT_LEVELS
    dataset_dir, output_dir = resolve_dataset_paths(
        args.variant,
        args.dataset_dir,
        args.output_dir,
    )

    generators = {
        "level_1": generate_and_write_level_1,
        "level_2": generate_and_write_level_2,
        "level_3": generate_and_write_level_3,
    }

    for level in levels:
        if level not in generators:
            raise NotImplementedError(f"Generation for {level} is not implemented yet")

        output_path = output_dir / level
        records = generators[level](
            template_path=args.template_dir / f"{level}.csv",
            seed_path=args.seed_dir / f"{level}.txt",
            dataset_dir=dataset_dir,
            output_path=output_path,
        )
        print(f"{level}: generated {len(records):,} questions -> {output_path}")


if __name__ == "__main__":
    main()
