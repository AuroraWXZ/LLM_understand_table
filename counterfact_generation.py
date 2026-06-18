"""Generate counterfactual benchmark tables from the cleaned dataset directory.

The supported run reads ``dataset_clean/`` and writes ``dataset_counter/``.
Extract samples afterward from ``dataset_counter/`` with
``extract_sample_dataset.py``.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import random
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence, TypeVar


DEFAULT_INPUT_DIR = Path("dataset_clean")
DEFAULT_OUTPUT_DIR = Path("dataset_counter")
DEFAULT_SEED = 42
NULL_VALUE = "NULL"

TABLE_FILENAMES = {
    "players": "players.csv",
    "countries": "countries.csv",
    "clubs": "clubs.csv",
    "competitions": "competitions.csv",
    "games": "games.csv",
    "game_events": "game_events.csv",
    "appearances": "appearances.csv",
    "transfers": "transfers.csv",
}

CLUB_REQUIRED_COLUMNS = [
    "domestic_competition_id",
    "country_name",
    "stadium_name",
]

COUNTRY_REQUIRED_COLUMNS = [
    "country_name",
    "confederation",
    "capital_city",
    "continent",
]

PLAYER_REQUIRED_COLUMNS = [
    "country_of_birth",
    "city_of_birth",
    "country_of_citizenship",
    "date_of_birth",
    "foot",
    "height_in_cm",
]

FOOT_VALUES = ["left", "right", "both"]

T = TypeVar("T")


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load a CSV as strings while preserving column order."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = [{field: row.get(field, "") or "" for field in fieldnames} for row in reader]
    return fieldnames, rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    """Write CSV rows with the original column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require_columns(fieldnames: Sequence[str], required_columns: Iterable[str], table_name: str) -> None:
    """Fail early when a table does not contain the requested columns."""
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        available = ", ".join(fieldnames)
        raise ValueError(
            f"{table_name}: missing required columns {missing}. "
            f"Available columns: {available}"
        )


def ordered_unique(values: Iterable[T]) -> list[T]:
    """Return first-seen unique values."""
    unique: list[T] = []
    seen: set[T] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def is_empty_value(value: object) -> bool:
    """Return True for empty strings and tuples containing an empty field."""
    if isinstance(value, tuple):
        return any(is_empty_value(part) for part in value)
    return str(value).strip() == ""


def build_chain_mapping(values: Iterable[T], rng: random.Random) -> dict[T, T]:
    """Map each unique value to another still-unused value, avoiding itself.

    This implements the requested chain-style swap: each source value chooses
    from remaining unswapped values except its own value. If the final value
    would otherwise map to itself, one earlier assignment is repaired.
    """
    unique_values = ordered_unique(value for value in values if not is_empty_value(value))
    if len(unique_values) < 2:
        return {}

    available = list(unique_values)
    mapping: dict[T, T] = {}

    for value in unique_values:
        candidates = [candidate for candidate in available if candidate != value]
        if candidates:
            chosen = rng.choice(candidates)
            mapping[value] = chosen
            available.remove(chosen)
            continue

        leftover = available[0]
        repairable_sources = [
            source for source, target in mapping.items() if target != value and source != leftover
        ]
        if not repairable_sources:
            raise RuntimeError("Could not build a valid chain mapping")

        source_to_repair = rng.choice(repairable_sources)
        mapping[value] = mapping[source_to_repair]
        mapping[source_to_repair] = leftover
        available.remove(leftover)

    return mapping


def apply_column_chain_swap(
    rows: list[dict[str, str]],
    column: str,
    rng: random.Random,
    row_indices: Iterable[int] | None = None,
) -> int:
    """Chain-swap one column and return the number of changed cells."""
    indices = list(range(len(rows))) if row_indices is None else list(row_indices)
    mapping = build_chain_mapping((rows[index].get(column, "") for index in indices), rng)
    changed = 0
    for index in indices:
        old_value = rows[index].get(column, "")
        if old_value not in mapping:
            continue
        rows[index][column] = mapping[old_value]
        changed += int(rows[index][column] != old_value)
    return changed


def apply_pair_chain_swap(
    rows: list[dict[str, str]],
    columns: tuple[str, str],
    rng: random.Random,
) -> int:
    """Chain-swap an aligned pair of columns and return changed cells."""
    values = ((row.get(columns[0], ""), row.get(columns[1], "")) for row in rows)
    mapping = build_chain_mapping(values, rng)
    changed = 0
    for row in rows:
        old_pair = (row.get(columns[0], ""), row.get(columns[1], ""))
        if old_pair not in mapping:
            continue
        new_pair = mapping[old_pair]
        row[columns[0]], row[columns[1]] = new_pair
        changed += int(new_pair[0] != old_pair[0]) + int(new_pair[1] != old_pair[1])
    return changed


def signed_delta(rng: random.Random, max_abs_delta: int) -> int:
    """Return +/- a random integer in [1, max_abs_delta]."""
    magnitude = rng.randint(1, max_abs_delta)
    return magnitude if rng.choice([True, False]) else -magnitude


def shift_month(month: int, delta: int) -> int:
    """Shift a 1-based month with circular wrapping."""
    return ((month - 1 + delta) % 12) + 1


def shift_birth_date(value: str, rng: random.Random) -> str:
    """Adjust birth year and month while keeping a valid ISO date."""
    if not value.strip():
        return value

    try:
        original = date.fromisoformat(value)
    except ValueError:
        return value

    new_year = original.year + signed_delta(rng, 5)
    new_month = shift_month(original.month, signed_delta(rng, 5))
    last_day = calendar.monthrange(new_year, new_month)[1]
    new_day = min(original.day, last_day)
    return date(new_year, new_month, new_day).isoformat()


def change_foot(value: str, rng: random.Random) -> str:
    """Pick one of the other canonical foot values."""
    normalized = value.strip().lower()
    if normalized not in FOOT_VALUES:
        return value
    return rng.choice([candidate for candidate in FOOT_VALUES if candidate != normalized])


def change_height(value: str, rng: random.Random) -> str:
    """Adjust height by +/- 1 to 3 cm."""
    if not value.strip():
        return value

    try:
        height = int(value)
    except ValueError:
        return value

    return str(max(1, height + signed_delta(rng, 3)))


def modify_clubs(rows: list[dict[str, str]], fieldnames: Sequence[str], rng: random.Random) -> dict[str, int]:
    """Modify clubs with aligned domestic competition/country swaps."""
    require_columns(fieldnames, CLUB_REQUIRED_COLUMNS, "clubs")
    return {
        "domestic_competition_country_cells": apply_pair_chain_swap(
            rows,
            ("domestic_competition_id", "country_name"),
            rng,
        ),
        "stadium_name_cells": apply_column_chain_swap(rows, "stadium_name", rng),
    }


def modify_countries(
    rows: list[dict[str, str]],
    fieldnames: Sequence[str],
    rng: random.Random,
) -> dict[str, int]:
    """Modify selected country attributes."""
    require_columns(fieldnames, COUNTRY_REQUIRED_COLUMNS, "countries")
    confederation_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("confederation", "").strip().casefold()
        not in {"", NULL_VALUE.casefold()}
    ]
    return {
        "confederation_cells": apply_column_chain_swap(
            rows,
            "confederation",
            rng,
            confederation_indices,
        ),
        "capital_city_cells": apply_column_chain_swap(rows, "capital_city", rng),
        "continent_cells": apply_column_chain_swap(rows, "continent", rng),
    }


def build_capital_to_country(countries: Sequence[dict[str, str]]) -> dict[str, str]:
    """Build a capital-city lookup after country table modification."""
    lookup: dict[str, str] = {}
    for country in countries:
        capital_city = country.get("capital_city", "").strip()
        country_name = country.get("country_name", "").strip()
        if capital_city and country_name and capital_city not in lookup:
            lookup[capital_city] = country_name
    return lookup


def modify_players(
    rows: list[dict[str, str]],
    fieldnames: Sequence[str],
    countries: Sequence[dict[str, str]],
    rng: random.Random,
) -> dict[str, int]:
    """Modify selected player attributes using the already-modified countries."""
    require_columns(fieldnames, PLAYER_REQUIRED_COLUMNS, "players")

    changed_counts = {
        "country_of_birth_cells": 0,
        "city_of_birth_cells": 0,
        "country_of_citizenship_cells": 0,
        "date_of_birth_cells": 0,
        "foot_cells": 0,
        "height_in_cm_cells": 0,
    }

    capital_to_country = build_capital_to_country(countries)
    non_capital_birth_indices: list[int] = []

    for index, row in enumerate(rows):
        city_of_birth = row.get("city_of_birth", "").strip()
        matching_country = capital_to_country.get(city_of_birth)
        if matching_country:
            old_country = row.get("country_of_birth", "")
            row["country_of_birth"] = matching_country
            changed_counts["country_of_birth_cells"] += int(matching_country != old_country)
        else:
            non_capital_birth_indices.append(index)

    changed_counts["country_of_birth_cells"] += apply_column_chain_swap(
        rows,
        "country_of_birth",
        rng,
        non_capital_birth_indices,
    )
    changed_counts["city_of_birth_cells"] = apply_column_chain_swap(rows, "city_of_birth", rng)
    changed_counts["country_of_citizenship_cells"] = apply_column_chain_swap(
        rows,
        "country_of_citizenship",
        rng,
    )

    for row in rows:
        old_date = row.get("date_of_birth", "")
        row["date_of_birth"] = shift_birth_date(old_date, rng)
        changed_counts["date_of_birth_cells"] += int(row["date_of_birth"] != old_date)

        old_foot = row.get("foot", "")
        row["foot"] = change_foot(old_foot, rng)
        changed_counts["foot_cells"] += int(row["foot"] != old_foot)

        old_height = row.get("height_in_cm", "")
        row["height_in_cm"] = change_height(old_height, rng)
        changed_counts["height_in_cm_cells"] += int(row["height_in_cm"] != old_height)

    return changed_counts


def validate_input_dir(input_dir: Path) -> None:
    """Restrict counterfactual generation to the cleaned full dataset."""
    if input_dir.name != DEFAULT_INPUT_DIR.name:
        raise ValueError(
            f"counterfact_generation only supports {DEFAULT_INPUT_DIR}/ as input. "
            f"Received: {input_dir}. Generate {DEFAULT_OUTPUT_DIR}/ from "
            f"{DEFAULT_INPUT_DIR}/, then extract samples from {DEFAULT_OUTPUT_DIR}/."
        )


def resolve_output_dir(output_dir: Path | None) -> Path:
    """Choose the requested default output directory."""
    if output_dir is not None:
        return output_dir
    return DEFAULT_OUTPUT_DIR


def csv_paths(input_dir: Path) -> list[Path]:
    """Return known CSV files first, then any extra CSVs in stable order."""
    known_paths = [input_dir / filename for filename in TABLE_FILENAMES.values()]
    extras = sorted(
        path
        for path in input_dir.glob("*.csv")
        if path.name not in set(TABLE_FILENAMES.values())
    )
    return [path for path in known_paths if path.exists()] + extras


def generate_counterfacts(input_dir: Path, output_dir: Path, seed: int) -> dict[str, dict[str, int]]:
    """Generate the counterfactual dataset and return per-table change counts."""
    validate_input_dir(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    rng = random.Random(seed)
    loaded_tables: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    summary: dict[str, dict[str, int]] = {}

    for path in csv_paths(input_dir):
        loaded_tables[path.name] = load_csv(path)

    if "countries.csv" in loaded_tables:
        fieldnames, rows = loaded_tables["countries.csv"]
        summary["countries"] = modify_countries(rows, fieldnames, rng)

    if "clubs.csv" in loaded_tables:
        fieldnames, rows = loaded_tables["clubs.csv"]
        summary["clubs"] = modify_clubs(rows, fieldnames, rng)

    if "players.csv" in loaded_tables:
        if "countries.csv" not in loaded_tables:
            raise FileNotFoundError("players.csv modification requires countries.csv")
        fieldnames, rows = loaded_tables["players.csv"]
        _, countries = loaded_tables["countries.csv"]
        summary["players"] = modify_players(rows, fieldnames, countries, rng)

    for filename, (fieldnames, rows) in loaded_tables.items():
        write_csv(output_dir / filename, fieldnames, rows)

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate counterfactual CSV tables from dataset_clean."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing source CSV files. Defaults to dataset_clean/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where counterfactual CSV files are written. Defaults to "
            "dataset_counter/."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible swaps. Defaults to {DEFAULT_SEED}.",
    )
    return parser.parse_args()


def main() -> None:
    """Run counterfactual generation."""
    args = parse_args()
    input_dir = args.input_dir
    output_dir = resolve_output_dir(args.output_dir)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {args.seed}")

    summary = generate_counterfacts(input_dir, output_dir, args.seed)
    for table_name, counts in summary.items():
        changed = sum(counts.values())
        details = ", ".join(f"{name}={count}" for name, count in counts.items())
        print(f"{table_name}: {changed} changed cells ({details})")

    print("\nDone.")


if __name__ == "__main__":
    main()
