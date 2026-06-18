"""Generate level 1 benchmark questions with answers and provenance."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path


DEFAULT_TEMPLATE_PATH = Path("question_template/level_1.csv")
DEFAULT_SEED_PATH = Path("question_template/seed/level_1.txt")
DEFAULT_DATASET_DIR = Path("dataset_clean")
DEFAULT_OUTPUT_PATH = Path("questions/level_1")

TABLE_ID_COLUMNS = {
    "players": "player_id",
    "countries": "country_id",
    "clubs": "club_id",
}

TABLE_FILES = {
    "players": "players.csv",
    "countries": "countries.csv",
    "clubs": "clubs.csv",
}

PLACEHOLDER_RE = re.compile(r"{(?P<name>[^{}]+)}")


@dataclass(frozen=True)
class Level1Context:
    """Template and table rows needed to generate level 1 questions."""

    templates: dict[int, dict[str, str]]
    table_rows: dict[str, dict[str, dict[str, str]]]

    @classmethod
    def load(
        cls,
        template_path: Path = DEFAULT_TEMPLATE_PATH,
        dataset_dir: Path = DEFAULT_DATASET_DIR,
    ) -> "Level1Context":
        templates = {
            int(row["question_id"]): row
            for row in _read_csv_rows(template_path)
        }
        table_rows = {
            table: _index_rows_by_id(_read_csv_rows(dataset_dir / filename), table)
            for table, filename in TABLE_FILES.items()
        }
        return cls(templates=templates, table_rows=table_rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _index_rows_by_id(rows: list[dict[str, str]], table: str) -> dict[str, dict[str, str]]:
    id_column = TABLE_ID_COLUMNS[table]
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = row.get(id_column, "")
        if not row_id:
            raise ValueError(f"{table}: row is missing {id_column}: {row}")
        indexed[str(row_id)] = row
    return indexed


def _split_csv_line(line: str) -> list[str]:
    return next(csv.reader([line], skipinitialspace=True))


def _strip_optional_braces(value: str) -> str:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        return value[1:-1].strip()
    return value


def _parse_key_value_row(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in _split_csv_line(line):
        if ":" not in part:
            raise ValueError(f"Expected key: value pair, got {part!r}")
        key, value = part.split(":", 1)
        key = _strip_optional_braces(key)
        if not key:
            raise ValueError(f"Missing seed key in row: {line!r}")
        values[key] = value.strip()
    return values


def _normalize_tables_key(value: str) -> str:
    return "; ".join(
        part.strip()
        for part in value.replace("+", ";").split(";")
        if part.strip()
    )


def parse_seed_file(path: Path = DEFAULT_SEED_PATH) -> dict[str, list[dict[str, str]]]:
    """Parse seed rows grouped by table section."""
    if not path.exists():
        raise FileNotFoundError(f"Missing seed file: {path}")

    grouped_seeds: dict[str, list[dict[str, str]]] = defaultdict(list)
    current_tables: str | None = None
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_tables = _normalize_tables_key(line.lstrip("#").strip())
            if current_tables not in TABLE_ID_COLUMNS:
                raise ValueError(f"{path}:{line_number}: unsupported level 1 table {current_tables!r}")
            continue
        if current_tables is None:
            raise ValueError(f"{path}:{line_number}: seed row found before a table section")
        grouped_seeds[current_tables].append(_parse_key_value_row(line))
    return dict(grouped_seeds)


def _json_scalar(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def _row_identity(table: str, row: dict[str, str]) -> dict[str, int | str]:
    id_attribute = TABLE_ID_COLUMNS[table]
    return {
        "id_attribute": id_attribute,
        "id_value": _json_scalar(row[id_attribute]),
    }


def _seed_column(table: str, key: str) -> str:
    key = _strip_optional_braces(key)
    if "." in key:
        table_name, column = key.split(".", 1)
        if table_name != table:
            raise ValueError(f"Seed key {key!r} does not belong to table {table!r}")
        key = column
    if table == "players" and key == "player_name":
        return "name"
    if table == "clubs" and key == "club_name":
        return "name"
    return key


def _resolve_seed_row(context: Level1Context, table: str, seed: dict[str, str]) -> dict[str, str]:
    id_column = TABLE_ID_COLUMNS[table]
    for key in (id_column, f"{table}.{id_column}"):
        if key in seed:
            row = context.table_rows[table].get(str(seed[key]))
            if row is None:
                raise ValueError(f"No {table} row found for {id_column}={seed[key]!r}")
            return row

    candidates = list(context.table_rows[table].values())
    for key, value in seed.items():
        column = _seed_column(table, key)
        candidates = [row for row in candidates if row.get(column) == value]
    if len(candidates) != 1:
        raise ValueError(
            f"Seed {seed!r} matched {len(candidates)} {table} rows; use the id column to disambiguate"
        )
    return candidates[0]


def _placeholder_names(template: str) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for match in PLACEHOLDER_RE.finditer(template):
        name = match.group("name").strip()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _placeholder_source(default_table: str, placeholder: str) -> tuple[str, str]:
    placeholder = placeholder.strip()
    if "." in placeholder:
        table, column = placeholder.split(".", 1)
    else:
        table, column = default_table, placeholder

    if table == "players" and column == "player_name":
        column = "name"
    if table == "clubs" and column == "club_name":
        column = "name"
    if table not in TABLE_ID_COLUMNS:
        raise ValueError(f"Unsupported placeholder table {table!r} in {placeholder!r}")
    return table, column


def _placeholder_values(
    table: str,
    row: dict[str, str],
    template: str,
) -> tuple[dict[str, str], dict[str, str]]:
    fill_values: dict[str, str] = {}
    provenance_values: dict[str, str] = {}
    for placeholder in _placeholder_names(template):
        source_table, column = _placeholder_source(table, placeholder)
        if source_table != table:
            raise ValueError(
                f"Level 1 template for {table!r} cannot use placeholder {placeholder!r}"
            )
        if column not in row:
            raise KeyError(f"{table} row has no column {column!r}")
        value = row[column]
        fill_values[placeholder] = value
        provenance_values[f"{source_table}.{column}"] = value
    return fill_values, provenance_values


def _fill_template(template: str, values: dict[str, str]) -> str:
    missing = [name for name in _placeholder_names(template) if name not in values]
    if missing:
        raise KeyError(f"Missing placeholder values for {', '.join(missing)} in {template!r}")
    return PLACEHOLDER_RE.sub(lambda match: values[match.group("name").strip()], template)


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _field_answer(attribute: str) -> Callable[[dict[str, str]], str]:
    return lambda row: row[attribute]


def _date_of_birth(row: dict[str, str]) -> date:
    return date.fromisoformat(row["date_of_birth"])


def _birth_year(row: dict[str, str]) -> str:
    return str(_date_of_birth(row).year)


def _birth_month(row: dict[str, str]) -> str:
    return _date_of_birth(row).strftime("%B")


def _birth_day(row: dict[str, str]) -> str:
    return str(_date_of_birth(row).day)


def _age_on(target: date) -> Callable[[dict[str, str]], str]:
    def answer(row: dict[str, str]) -> str:
        born = _date_of_birth(row)
        age = target.year - born.year - ((target.month, target.day) < (born.month, born.day))
        return str(age)

    return answer


def _born_before(year: int) -> Callable[[dict[str, str]], str]:
    return lambda row: _yes_no(_date_of_birth(row).year < year)


def _born_in_or_after(year: int) -> Callable[[dict[str, str]], str]:
    return lambda row: _yes_no(_date_of_birth(row).year >= year)


def _born_between(start_year: int, end_year: int) -> Callable[[dict[str, str]], str]:
    return lambda row: _yes_no(start_year <= _date_of_birth(row).year <= end_year)


def _born_month(month: int) -> Callable[[dict[str, str]], str]:
    return lambda row: _yes_no(_date_of_birth(row).month == month)


def _height_at_least(value: int) -> Callable[[dict[str, str]], str]:
    return lambda row: _yes_no(int(row["height_in_cm"]) >= value)


def _height_shorter_than(value: int) -> Callable[[dict[str, str]], str]:
    return lambda row: _yes_no(int(row["height_in_cm"]) < value)


def _height_taller_than(value: int) -> Callable[[dict[str, str]], str]:
    return lambda row: _yes_no(int(row["height_in_cm"]) > value)


def _field_equals(attribute: str, expected: str) -> Callable[[dict[str, str]], str]:
    return lambda row: _yes_no(row[attribute].casefold() == expected.casefold())


def _values_match(actual: str, expected: str) -> bool:
    return actual.strip().casefold() == expected.strip().casefold()


def _field_not_equals(attribute: str, expected: str) -> Callable[[dict[str, str]], str]:
    return lambda row: _yes_no(not _values_match(row[attribute], expected))


def _single_row_question(
    context: Level1Context,
    template_id: int,
    seed: dict[str, str],
    table: str,
    answer_attribute: str,
    answer_func: Callable[[dict[str, str]], str] | None = None,
    expected_placeholder: str | None = None,
) -> dict[str, object]:
    template = context.templates[template_id]
    row = _resolve_seed_row(context, table, seed)
    fill_values, placeholder_provenance = _placeholder_values(table, row, template["question"])
    row_identity = _row_identity(table, row)
    if expected_placeholder is not None:
        if expected_placeholder not in fill_values:
            raise KeyError(
                f"Template {template_id} has no placeholder {expected_placeholder!r}"
            )
        answer = _yes_no(_values_match(row[answer_attribute], fill_values[expected_placeholder]))
    elif answer_func is not None:
        answer = str(answer_func(row))
    else:
        raise ValueError(f"Template {template_id} has no answer function")
    return {
        "template_question_id": template_id,
        "question": _fill_template(template["question"], fill_values),
        "answer": answer,
        "ground_truth": {
            "rows": [
                {
                    "table": table,
                    **row_identity,
                }
            ],
            "placeholders": placeholder_provenance,
            "answer_source": {
                "table": table,
                **row_identity,
                "attribute": answer_attribute,
            },
        },
    }

def question_0(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 0."""
    return _single_row_question(
        context,
        0,
        seed,
        "players",
        "country_of_birth",
        _field_answer("country_of_birth"),
    )

def question_1(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 1."""
    return _single_row_question(
        context,
        1,
        seed,
        "players",
        "city_of_birth",
        _field_answer("city_of_birth"),
    )

def question_2(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 2."""
    return _single_row_question(
        context,
        2,
        seed,
        "players",
        "country_of_citizenship",
        _field_answer("country_of_citizenship"),
    )

def question_3(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 3."""
    return _single_row_question(
        context,
        3,
        seed,
        "players",
        "date_of_birth",
        _field_answer("date_of_birth"),
    )

def question_4(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 4."""
    return _single_row_question(
        context,
        4,
        seed,
        "players",
        "foot",
        _field_answer("foot"),
    )

def question_5(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 5."""
    return _single_row_question(
        context,
        5,
        seed,
        "players",
        "height_in_cm",
        _field_answer("height_in_cm"),
    )

def question_6(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 6."""
    return _single_row_question(
        context,
        6,
        seed,
        "players",
        "country_of_birth",
        expected_placeholder="country_of_birth",
    )

def question_7(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 7."""
    return _single_row_question(
        context,
        7,
        seed,
        "players",
        "city_of_birth",
        expected_placeholder="city_of_birth",
    )

def question_8(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 8."""
    return _single_row_question(
        context,
        8,
        seed,
        "players",
        "country_of_citizenship",
        expected_placeholder="country_of_citizenship",
    )

def question_9(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 9."""
    return _single_row_question(
        context,
        9,
        seed,
        "players",
        "foot",
        expected_placeholder="foot",
    )

def question_10(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 10."""
    return _single_row_question(
        context,
        10,
        seed,
        "players",
        "foot",
        _field_equals("foot", "right"),
    )

def question_11(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 11."""
    return _single_row_question(
        context,
        11,
        seed,
        "players",
        "foot",
        _field_equals("foot", "left"),
    )

def question_12(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 12."""
    return _single_row_question(
        context,
        12,
        seed,
        "players",
        "date_of_birth",
        _birth_year,
    )

def question_13(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 13."""
    return _single_row_question(
        context,
        13,
        seed,
        "players",
        "date_of_birth",
        _birth_month,
    )

def question_14(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 14."""
    return _single_row_question(
        context,
        14,
        seed,
        "players",
        "date_of_birth",
        _birth_day,
    )

def question_15(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 15."""
    return _single_row_question(
        context,
        15,
        seed,
        "players",
        "date_of_birth",
        _age_on(date(2025, 12, 31)),
    )

def question_16(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 16."""
    return _single_row_question(
        context,
        16,
        seed,
        "players",
        "date_of_birth",
        _born_before(2000),
    )

def question_17(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 17."""
    return _single_row_question(
        context,
        17,
        seed,
        "players",
        "date_of_birth",
        _born_in_or_after(2000),
    )

def question_18(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 18."""
    return _single_row_question(
        context,
        18,
        seed,
        "players",
        "date_of_birth",
        _born_before(1995),
    )

def question_19(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 19."""
    return _single_row_question(
        context,
        19,
        seed,
        "players",
        "date_of_birth",
        _born_in_or_after(1995),
    )

def question_20(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 20."""
    return _single_row_question(
        context,
        20,
        seed,
        "players",
        "date_of_birth",
        _born_between(1990, 1999),
    )

def question_21(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 21."""
    return _single_row_question(
        context,
        21,
        seed,
        "players",
        "date_of_birth",
        _born_between(2000, 2009),
    )

def question_22(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 22."""
    return _single_row_question(
        context,
        22,
        seed,
        "players",
        "date_of_birth",
        _born_month(1),
    )

def question_23(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 23."""
    return _single_row_question(
        context,
        23,
        seed,
        "players",
        "date_of_birth",
        _born_month(6),
    )

def question_24(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 24."""
    return _single_row_question(
        context,
        24,
        seed,
        "players",
        "date_of_birth",
        _born_month(12),
    )

def question_25(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 25."""
    return _single_row_question(
        context,
        25,
        seed,
        "players",
        "height_in_cm",
        _height_at_least(180),
    )

def question_26(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 26."""
    return _single_row_question(
        context,
        26,
        seed,
        "players",
        "height_in_cm",
        _height_shorter_than(175),
    )

def question_27(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 27."""
    return _single_row_question(
        context,
        27,
        seed,
        "players",
        "height_in_cm",
        _height_taller_than(185),
    )

def question_28(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 28."""
    return _single_row_question(
        context,
        28,
        seed,
        "players",
        "height_in_cm",
        _height_at_least(190),
    )

def question_29(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 29."""
    return _single_row_question(
        context,
        29,
        seed,
        "players",
        "height_in_cm",
        _height_shorter_than(170),
    )

def question_30(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 30."""
    return _single_row_question(
        context,
        30,
        seed,
        "players",
        "height_in_cm",
        expected_placeholder="height_in_cm",
    )

def question_31(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 31."""
    return _single_row_question(
        context,
        31,
        seed,
        "countries",
        "capital_city",
        _field_answer("capital_city"),
    )

def question_32(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 32."""
    return _single_row_question(
        context,
        32,
        seed,
        "countries",
        "continent",
        _field_answer("continent"),
    )

def question_33(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 33."""
    return _single_row_question(
        context,
        33,
        seed,
        "countries",
        "confederation",
        _field_answer("confederation"),
    )

def question_34(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 34."""
    return _single_row_question(
        context,
        34,
        seed,
        "countries",
        "continent",
        _field_equals("continent", "Europe"),
    )

def question_35(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 35."""
    return _single_row_question(
        context,
        35,
        seed,
        "countries",
        "continent",
        _field_equals("continent", "Asia"),
    )

def question_36(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 36."""
    return _single_row_question(
        context,
        36,
        seed,
        "countries",
        "continent",
        _field_equals("continent", "Africa"),
    )

def question_37(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 37."""
    return _single_row_question(
        context,
        37,
        seed,
        "countries",
        "continent",
        _field_equals("continent", "North America"),
    )

def question_38(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 38."""
    return _single_row_question(
        context,
        38,
        seed,
        "countries",
        "continent",
        _field_equals("continent", "South America"),
    )

def question_39(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 39."""
    return _single_row_question(
        context,
        39,
        seed,
        "countries",
        "continent",
        _field_equals("continent", "Oceania"),
    )

def question_40(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 40."""
    return _single_row_question(
        context,
        40,
        seed,
        "countries",
        "confederation",
        _field_equals("confederation", "europa"),
    )

def question_41(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 41."""
    return _single_row_question(
        context,
        41,
        seed,
        "countries",
        "confederation",
        _field_equals("confederation", "asien"),
    )

def question_42(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 42."""
    return _single_row_question(
        context,
        42,
        seed,
        "countries",
        "confederation",
        _field_equals("confederation", "afrika"),
    )

def question_43(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 43."""
    return _single_row_question(
        context,
        43,
        seed,
        "countries",
        "confederation",
        _field_equals("confederation", "amerika"),
    )

def question_44(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 44."""
    return _single_row_question(
        context,
        44,
        seed,
        "countries",
        "confederation",
        _field_equals("confederation", "NULL"),
    )

def question_45(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 45."""
    return _single_row_question(
        context,
        45,
        seed,
        "countries",
        "confederation",
        _field_not_equals("confederation", "NULL"),
    )

def question_46(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 46."""
    return _single_row_question(
        context,
        46,
        seed,
        "countries",
        "capital_city",
        expected_placeholder="capital_city",
    )

def question_47(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 47."""
    return _single_row_question(
        context,
        47,
        seed,
        "countries",
        "continent",
        expected_placeholder="continent",
    )

def question_48(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 48."""
    return _single_row_question(
        context,
        48,
        seed,
        "countries",
        "confederation",
        expected_placeholder="confederation",
    )

def question_49(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 49."""
    return _single_row_question(
        context,
        49,
        seed,
        "clubs",
        "country_name",
        _field_answer("country_name"),
    )

def question_50(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 50."""
    return _single_row_question(
        context,
        50,
        seed,
        "clubs",
        "stadium_name",
        _field_answer("stadium_name"),
    )

def question_51(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 51."""
    return _single_row_question(
        context,
        51,
        seed,
        "clubs",
        "domestic_competition_id",
        _field_answer("domestic_competition_id"),
    )

def question_52(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 52."""
    return _single_row_question(
        context,
        52,
        seed,
        "clubs",
        "country_name",
        expected_placeholder="country_name",
    )

def question_53(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 53."""
    return _single_row_question(
        context,
        53,
        seed,
        "clubs",
        "stadium_name",
        expected_placeholder="stadium_name",
    )

def question_54(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 54."""
    return _single_row_question(
        context,
        54,
        seed,
        "clubs",
        "domestic_competition_id",
        expected_placeholder="domestic_competition_id",
    )

def question_55(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 55."""
    return _single_row_question(
        context,
        55,
        seed,
        "clubs",
        "country_name",
        _field_answer("country_name"),
    )

def question_56(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 56."""
    return _single_row_question(
        context,
        56,
        seed,
        "clubs",
        "stadium_name",
        _field_answer("stadium_name"),
    )

def question_57(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 57."""
    return _single_row_question(
        context,
        57,
        seed,
        "clubs",
        "domestic_competition_id",
        _field_answer("domestic_competition_id"),
    )

def question_58(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 58."""
    return _single_row_question(
        context,
        58,
        seed,
        "clubs",
        "stadium_name",
        expected_placeholder="stadium_name",
    )

def question_59(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 59."""
    return _single_row_question(
        context,
        59,
        seed,
        "clubs",
        "country_name",
        expected_placeholder="country_name",
    )

def question_60(context: Level1Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 1 template question 60."""
    return _single_row_question(
        context,
        60,
        seed,
        "clubs",
        "domestic_competition_id",
        expected_placeholder="domestic_competition_id",
    )

QUESTION_FUNCTIONS = {
    0: question_0,
    1: question_1,
    2: question_2,
    3: question_3,
    4: question_4,
    5: question_5,
    6: question_6,
    7: question_7,
    8: question_8,
    9: question_9,
    10: question_10,
    11: question_11,
    12: question_12,
    13: question_13,
    14: question_14,
    15: question_15,
    16: question_16,
    17: question_17,
    18: question_18,
    19: question_19,
    20: question_20,
    21: question_21,
    22: question_22,
    23: question_23,
    24: question_24,
    25: question_25,
    26: question_26,
    27: question_27,
    28: question_28,
    29: question_29,
    30: question_30,
    31: question_31,
    32: question_32,
    33: question_33,
    34: question_34,
    35: question_35,
    36: question_36,
    37: question_37,
    38: question_38,
    39: question_39,
    40: question_40,
    41: question_41,
    42: question_42,
    43: question_43,
    44: question_44,
    45: question_45,
    46: question_46,
    47: question_47,
    48: question_48,
    49: question_49,
    50: question_50,
    51: question_51,
    52: question_52,
    53: question_53,
    54: question_54,
    55: question_55,
    56: question_56,
    57: question_57,
    58: question_58,
    59: question_59,
    60: question_60,
}


def generate_level_1_questions(
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    seed_path: Path = DEFAULT_SEED_PATH,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> list[dict[str, object]]:
    """Generate all level 1 questions from seed rows."""
    context = Level1Context.load(template_path=template_path, dataset_dir=dataset_dir)
    seeds_by_table = parse_seed_file(seed_path)

    records: list[dict[str, object]] = []
    for template_id in sorted(context.templates):
        template = context.templates[template_id]
        table = _normalize_tables_key(template["tables"])
        if table not in seeds_by_table:
            continue
        if template_id not in QUESTION_FUNCTIONS:
            raise KeyError(f"No level 1 generator function for template {template_id}")
        for seed in seeds_by_table[table]:
            record = QUESTION_FUNCTIONS[template_id](context, seed)
            records.append({"question_id": len(records), **record})
    return records


def write_questions_json(records: list[dict[str, object]], output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Write generated benchmark records as a JSON array."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def generate_and_write_level_1(
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    seed_path: Path = DEFAULT_SEED_PATH,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, object]]:
    """Generate level 1 records and write them to disk."""
    records = generate_level_1_questions(
        template_path=template_path,
        seed_path=seed_path,
        dataset_dir=dataset_dir,
    )
    write_questions_json(records, output_path=output_path)
    return records
