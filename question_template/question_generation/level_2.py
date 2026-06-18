"""Generate level 2 benchmark questions with answers and provenance."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path


DEFAULT_TEMPLATE_PATH = Path("question_template/level_2.csv")
DEFAULT_SEED_PATH = Path("question_template/seed/level_2.txt")
DEFAULT_DATASET_DIR = Path("dataset_clean")
DEFAULT_OUTPUT_PATH = Path("questions/level_2")

TABLE_ID_COLUMNS = {
    "players": "player_id",
    "countries": "country_id",
    "clubs": "club_id",
    "competitions": "competition_id",
    "appearance": "appearance_id",
    "games": "game_id",
    "game_events": "game_event_id",
}

TABLE_FILES = {
    "players": "players.csv",
    "countries": "countries.csv",
    "clubs": "clubs.csv",
    "competitions": "competitions.csv",
    "appearance": "appearances.csv",
    "games": "games.csv",
    "game_events": "game_events.csv",
}

PLACEHOLDER_RE = re.compile(r"{(?P<name>[^{}]+)}")


@dataclass(frozen=True)
class Level2Context:
    """Template and table rows needed to generate level 2 questions."""

    templates: dict[int, dict[str, str]]
    table_rows: dict[str, dict[str, dict[str, str]]]
    rows_by_name: dict[str, dict[str, dict[str, str]]]
    game_events_by_game_id: dict[str, list[dict[str, str]]]

    @classmethod
    def load(
        cls,
        template_path: Path = DEFAULT_TEMPLATE_PATH,
        dataset_dir: Path = DEFAULT_DATASET_DIR,
    ) -> "Level2Context":
        templates = {
            int(row["question_id"]): row
            for row in _read_csv_rows(template_path)
        }
        table_rows = {
            table: _index_rows_by_id(_read_csv_rows(dataset_dir / filename), table)
            for table, filename in TABLE_FILES.items()
        }
        rows_by_name = {
            "countries": {row["country_name"]: row for row in table_rows["countries"].values()},
            "clubs": {row["name"]: row for row in table_rows["clubs"].values()},
        }
        game_events_by_game_id: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in table_rows["game_events"].values():
            game_events_by_game_id[row["game_id"]].append(row)
        for rows in game_events_by_game_id.values():
            rows.sort(key=lambda row: (int(row["minute"] or 0), row["game_event_id"]))
        return cls(
            templates=templates,
            table_rows=table_rows,
            rows_by_name=rows_by_name,
            game_events_by_game_id=dict(game_events_by_game_id),
        )


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


def _resolve_seed_row(context: Level2Context, table: str, seed: dict[str, str]) -> dict[str, str]:
    id_column = TABLE_ID_COLUMNS[table]
    for key in (id_column, f"{table}.{id_column}"):
        if key not in seed:
            continue
        value = str(seed[key])
        row = context.table_rows[table].get(value)
        if row is not None:
            return row
        if table == "game_events":
            events_for_game = context.game_events_by_game_id.get(value, [])
            if events_for_game:
                return events_for_game[0]
        raise ValueError(f"No {table} row found for {id_column}={value!r}")

    candidates = list(context.table_rows[table].values())
    for key, value in seed.items():
        if key.split(".", 1)[0] != table and "." in key:
            continue
        column = _seed_column(table, key)
        candidates = [row for row in candidates if row.get(column) == value]
    if len(candidates) != 1:
        raise ValueError(
            f"Seed {seed!r} matched {len(candidates)} {table} rows; use the id column to disambiguate"
        )
    return candidates[0]


def _row_ref(table: str, row: dict[str, str]) -> dict[str, int | str]:
    return {
        "table": table,
        **_row_identity(table, row),
    }


def _dedupe_row_refs(rows: list[tuple[str, dict[str, str]]]) -> list[dict[str, int | str]]:
    seen: set[tuple[str, int | str]] = set()
    refs: list[dict[str, int | str]] = []
    for table, row in rows:
        ref = _row_ref(table, row)
        key = (ref["table"], ref["id_attribute"], ref["id_value"])
        if key not in seen:
            seen.add(key)
            refs.append(ref)
    return refs


def _answer_source(table: str, row: dict[str, str], attribute: str) -> dict[str, int | str]:
    return {
        "table": table,
        **_row_identity(table, row),
        "attribute": attribute,
    }


def _template_attribute_paths(context: Level2Context, template_id: int) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for raw_path in context.templates[template_id]["attributes"].split(";"):
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        if "." not in raw_path:
            raise ValueError(f"Template {template_id}: invalid attribute path {raw_path!r}")
        table, attribute = raw_path.split(".", 1)
        paths.append((table.strip(), attribute.strip()))
    if not paths:
        raise ValueError(f"Template {template_id}: no attributes configured")
    return paths


def _template_tables(context: Level2Context, template_id: int) -> set[str]:
    return {
        part.strip()
        for part in context.templates[template_id]["tables"].replace("+", ";").split(";")
        if part.strip()
    }


def _answer_path(context: Level2Context, template_id: int) -> tuple[str, str]:
    return _template_attribute_paths(context, template_id)[-1]


def _answer_attribute(
    context: Level2Context,
    template_id: int,
    expected_table: str,
) -> str:
    table, attribute = _answer_path(context, template_id)
    if table != expected_table:
        raise ValueError(
            f"Template {template_id}: expected answer from {expected_table}, "
            f"but metadata ends with {table}.{attribute}"
        )
    return attribute


def _first_template_attribute(
    context: Level2Context,
    template_id: int,
    table: str,
    candidates: set[str],
) -> str:
    for path_table, attribute in _template_attribute_paths(context, template_id):
        if path_table == table and attribute in candidates:
            return attribute
    raise ValueError(
        f"Template {template_id}: none of {sorted(candidates)} found for table {table!r}"
    )


def _validate_record_against_template(
    context: Level2Context,
    template_id: int,
    record: dict[str, object],
) -> None:
    template_tables = _template_tables(context, template_id)
    ground_truth = record["ground_truth"]
    rows = ground_truth["rows"]
    row_tables = {row["table"] for row in rows}
    extra_tables = row_tables - template_tables
    if extra_tables:
        raise ValueError(
            f"Template {template_id}: generated ground-truth tables {sorted(row_tables)} "
            f"but template tables are {sorted(template_tables)}"
        )

    answer_source = ground_truth["answer_source"]
    source_path = f"{answer_source['table']}.{answer_source['attribute']}"
    template_attribute_paths = {
        f"{table}.{attribute}"
        for table, attribute in _template_attribute_paths(context, template_id)
    }
    if source_path not in template_attribute_paths:
        raise ValueError(
            f"Template {template_id}: answer source {source_path!r} is not listed in "
            f"template attributes {sorted(template_attribute_paths)}"
        )


def _placeholder_names(template: str) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for match in PLACEHOLDER_RE.finditer(template):
        name = match.group("name").strip()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _placeholder_source(
    placeholder: str,
    fill_rows: dict[str, dict[str, str]],
    default_table: str | None,
) -> tuple[str, str]:
    placeholder = placeholder.strip()
    if "." in placeholder:
        table, column = placeholder.split(".", 1)
    else:
        column = placeholder
        if default_table and column in fill_rows.get(default_table, {}):
            table = default_table
        else:
            candidate_tables = [
                table_name
                for table_name, row in fill_rows.items()
                if column in row
            ]
            if len(candidate_tables) != 1:
                raise KeyError(
                    f"Could not resolve unqualified placeholder {placeholder!r}; "
                    f"candidate tables: {candidate_tables}"
                )
            table = candidate_tables[0]

    if table == "players" and column == "player_name":
        column = "name"
    if table == "clubs" and column == "club_name":
        column = "name"
    if table not in fill_rows:
        raise KeyError(f"No row available for placeholder table {table!r}")
    return table, column


def _fill_template(
    template: str,
    fill_rows: dict[str, dict[str, str]],
    default_table: str | None,
) -> tuple[str, dict[str, str]]:
    fill_values: dict[str, str] = {}
    placeholder_provenance: dict[str, str] = {}
    for placeholder in _placeholder_names(template):
        table, column = _placeholder_source(placeholder, fill_rows, default_table)
        row = fill_rows[table]
        if column not in row:
            raise KeyError(f"{table} row has no column {column!r}")
        value = row[column]
        fill_values[placeholder] = value
        placeholder_provenance[f"{table}.{column}"] = value

    question = PLACEHOLDER_RE.sub(lambda match: fill_values[match.group("name").strip()], template)
    return question, placeholder_provenance


def _record(
    context: Level2Context,
    template_id: int,
    fill_rows: dict[str, dict[str, str]],
    ground_truth_rows: list[tuple[str, dict[str, str]]],
    answer: str,
    source_table: str,
    source_row: dict[str, str],
    source_attribute: str,
    default_table: str | None = None,
) -> dict[str, object]:
    template = context.templates[template_id]
    question, placeholders = _fill_template(template["question"], fill_rows, default_table)
    record: dict[str, object] = {
        "template_question_id": template_id,
        "question": question,
        "answer": str(answer),
        "ground_truth": {
            "rows": _dedupe_row_refs(ground_truth_rows),
            "placeholders": placeholders,
            "answer_source": _answer_source(source_table, source_row, source_attribute),
        },
    }
    _validate_record_against_template(context, template_id, record)
    return record


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _date_of_birth(row: dict[str, str]) -> date:
    return date.fromisoformat(row["date_of_birth"])


def _age_on(target: date) -> Callable[[dict[str, str]], str]:
    def answer(row: dict[str, str]) -> str:
        born = _date_of_birth(row)
        age = target.year - born.year - ((target.month, target.day) < (born.month, born.day))
        return str(age)

    return answer


def _country_by_name(context: Level2Context, country_name: str) -> dict[str, str]:
    row = context.rows_by_name["countries"].get(country_name)
    if row is None:
        raise ValueError(f"No country row found for country_name={country_name!r}")
    return row


def _country_for_competition(context: Level2Context, competition: dict[str, str]) -> dict[str, str]:
    country_id = competition.get("country_id", "")
    if country_id and country_id != "-1":
        row = context.table_rows["countries"].get(country_id)
        if row is None:
            raise ValueError(f"No country row found for competition country_id={country_id!r}")
        return row
    if competition.get("country_name"):
        return _country_by_name(context, competition["country_name"])
    raise ValueError(
        "Competition has no country_id or country_name that can be resolved: "
        f"{competition.get('competition_id', '')!r}"
    )


def _player_from_seed(context: Level2Context, seed: dict[str, str]) -> dict[str, str]:
    return _resolve_seed_row(context, "players", seed)


def _club_from_seed(context: Level2Context, seed: dict[str, str]) -> dict[str, str]:
    return _resolve_seed_row(context, "clubs", seed)


def _appearance_from_seed(context: Level2Context, seed: dict[str, str]) -> dict[str, str]:
    return _resolve_seed_row(context, "appearance", seed)


def _game_from_seed(context: Level2Context, seed: dict[str, str]) -> dict[str, str]:
    return _resolve_seed_row(context, "games", seed)


def _event_from_seed(context: Level2Context, seed: dict[str, str]) -> dict[str, str]:
    return _resolve_seed_row(context, "game_events", seed)


def _comparison_country_from_seed(
    context: Level2Context,
    seed: dict[str, str],
    fallback: dict[str, str],
) -> dict[str, str]:
    country_seed_keys = [key for key in seed if key == "country_id" or key.startswith("countries.")]
    if country_seed_keys:
        return _resolve_seed_row(context, "countries", seed)
    return fallback


def _competition_for_club(context: Level2Context, club: dict[str, str]) -> dict[str, str]:
    competition_id = club["domestic_competition_id"]
    row = context.table_rows["competitions"].get(competition_id)
    if row is None:
        raise ValueError(f"No competition row found for competition_id={competition_id!r}")
    return row


def _club_by_id(context: Level2Context, club_id: str) -> dict[str, str]:
    row = context.table_rows["clubs"].get(str(club_id))
    if row is None:
        raise ValueError(f"No club row found for club_id={club_id!r}")
    return row


def _player_by_id(context: Level2Context, player_id: str) -> dict[str, str]:
    row = context.table_rows["players"].get(str(player_id))
    if row is None:
        raise ValueError(f"No player row found for player_id={player_id!r}")
    return row


def _competition_by_id(context: Level2Context, competition_id: str) -> dict[str, str]:
    row = context.table_rows["competitions"].get(str(competition_id))
    if row is None:
        raise ValueError(f"No competition row found for competition_id={competition_id!r}")
    return row


def _player_country_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    player_country_attribute = _first_template_attribute(
        context,
        template_id,
        "players",
        {"country_of_birth", "country_of_citizenship"},
    )
    country_attribute = _answer_attribute(context, template_id, "countries")
    player = _player_from_seed(context, seed)
    country = _country_by_name(context, player[player_country_attribute])
    return _record(
        context,
        template_id,
        {"players": player, "countries": country},
        [("players", player), ("countries", country)],
        country[country_attribute],
        "countries",
        country,
        country_attribute,
        default_table="players",
    )


def _player_country_compare(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    country_attribute = _answer_attribute(context, template_id, "countries")
    player = _player_from_seed(context, seed)
    birth_country = _country_by_name(context, player["country_of_birth"])
    citizenship_country = _country_by_name(context, player["country_of_citizenship"])
    return _record(
        context,
        template_id,
        {"players": player},
        [("players", player), ("countries", birth_country), ("countries", citizenship_country)],
        _yes_no(birth_country[country_attribute] == citizenship_country[country_attribute]),
        "countries",
        birth_country,
        country_attribute,
        default_table="players",
    )


def _club_country_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    country_attribute = _answer_attribute(context, template_id, "countries")
    club = _club_from_seed(context, seed)
    country = _country_by_name(context, club["country_name"])
    return _record(
        context,
        template_id,
        {"clubs": club, "countries": country},
        [("clubs", club), ("countries", country)],
        country[country_attribute],
        "countries",
        country,
        country_attribute,
        default_table="clubs",
    )


def _club_competition_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    competition_attribute = _answer_attribute(context, template_id, "competitions")
    club = _club_from_seed(context, seed)
    competition = _competition_for_club(context, club)
    return _record(
        context,
        template_id,
        {"clubs": club, "competitions": competition},
        [("clubs", club), ("competitions", competition)],
        competition[competition_attribute],
        "competitions",
        competition,
        competition_attribute,
        default_table="clubs",
    )


def _club_competition_country_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    country_attribute = _answer_attribute(context, template_id, "countries")
    club = _club_from_seed(context, seed)
    competition = _competition_for_club(context, club)
    country = _country_for_competition(context, competition)
    return _record(
        context,
        template_id,
        {"clubs": club, "competitions": competition, "countries": country},
        [("clubs", club), ("competitions", competition), ("countries", country)],
        country[country_attribute],
        "countries",
        country,
        country_attribute,
        default_table="clubs",
    )


def _club_competition_country_compare(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    country_attribute = _answer_attribute(context, template_id, "countries")
    club = _club_from_seed(context, seed)
    competition = _competition_for_club(context, club)
    actual_country = _country_for_competition(context, competition)
    if template_id == 24:
        return _record(
            context,
            template_id,
            {"clubs": club, "competitions": competition, "countries": actual_country},
            [("clubs", club), ("competitions", competition), ("countries", actual_country)],
            _yes_no(actual_country[country_attribute] == "Europe"),
            "countries",
            actual_country,
            country_attribute,
            default_table="clubs",
        )
    comparison_country = _comparison_country_from_seed(context, seed, actual_country)
    return _record(
        context,
        template_id,
        {"clubs": club, "competitions": competition, "countries": comparison_country},
        [("clubs", club), ("competitions", competition), ("countries", actual_country), ("countries", comparison_country)],
        _yes_no(actual_country[country_attribute] == comparison_country[country_attribute]),
        "countries",
        actual_country,
        country_attribute,
        default_table="countries",
    )


def _appearance_player_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    player_attribute = _answer_attribute(context, template_id, "players")
    appearance = _appearance_from_seed(context, seed)
    player = _player_by_id(context, appearance["player_id"])
    return _record(
        context,
        template_id,
        {"appearance": appearance, "players": player},
        [("appearance", appearance), ("players", player)],
        player[player_attribute],
        "players",
        player,
        player_attribute,
        default_table="players",
    )


def _appearance_player_age(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    source_attribute = _answer_attribute(context, template_id, "players")
    appearance = _appearance_from_seed(context, seed)
    player = _player_by_id(context, appearance["player_id"])
    answer_func = _age_on(date(2025, 12, 31))
    return _record(
        context,
        template_id,
        {"appearance": appearance, "players": player},
        [("appearance", appearance), ("players", player)],
        answer_func(player),
        "players",
        player,
        source_attribute,
        default_table="players",
    )


def _appearance_player_country_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    player_country_attribute = _first_template_attribute(
        context,
        template_id,
        "players",
        {"country_of_birth", "country_of_citizenship"},
    )
    country_attribute = _answer_attribute(context, template_id, "countries")
    appearance = _appearance_from_seed(context, seed)
    player = _player_by_id(context, appearance["player_id"])
    country = _country_by_name(context, player[player_country_attribute])
    return _record(
        context,
        template_id,
        {"appearance": appearance, "players": player, "countries": country},
        [("appearance", appearance), ("players", player), ("countries", country)],
        country[country_attribute],
        "countries",
        country,
        country_attribute,
        default_table="players",
    )


def _appearance_club_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    club_id_attribute = _first_template_attribute(
        context,
        template_id,
        "appearance",
        {"player_club_id", "player_current_club_id"},
    )
    club_attribute = _answer_attribute(context, template_id, "clubs")
    appearance = _appearance_from_seed(context, seed)
    club = _club_by_id(context, appearance[club_id_attribute])
    return _record(
        context,
        template_id,
        {"appearance": appearance, "clubs": club},
        [("appearance", appearance), ("clubs", club)],
        club[club_attribute],
        "clubs",
        club,
        club_attribute,
        default_table="appearance",
    )


def _game_club_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    club_id_attribute = _first_template_attribute(
        context,
        template_id,
        "games",
        {"home_club_id", "away_club_id"},
    )
    club_attribute = _answer_attribute(context, template_id, "clubs")
    game = _game_from_seed(context, seed)
    club = _club_by_id(context, game[club_id_attribute])
    return _record(
        context,
        template_id,
        {"games": game, "clubs": club},
        [("games", game), ("clubs", club)],
        club[club_attribute],
        "clubs",
        club,
        club_attribute,
        default_table="games",
    )


def _game_club_compare(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    club_attribute = _answer_attribute(context, template_id, "clubs")
    game = _game_from_seed(context, seed)
    home_club = _club_by_id(context, game["home_club_id"])
    away_club = _club_by_id(context, game["away_club_id"])
    return _record(
        context,
        template_id,
        {"games": game},
        [("games", game), ("clubs", home_club), ("clubs", away_club)],
        _yes_no(home_club[club_attribute] == away_club[club_attribute]),
        "clubs",
        home_club,
        club_attribute,
        default_table="games",
    )


def _game_club_country_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    club_id_attribute = _first_template_attribute(
        context,
        template_id,
        "games",
        {"home_club_id", "away_club_id"},
    )
    country_attribute = _answer_attribute(context, template_id, "countries")
    game = _game_from_seed(context, seed)
    club = _club_by_id(context, game[club_id_attribute])
    country = _country_by_name(context, club["country_name"])
    return _record(
        context,
        template_id,
        {"games": game, "clubs": club, "countries": country},
        [("games", game), ("clubs", club), ("countries", country)],
        country[country_attribute],
        "countries",
        country,
        country_attribute,
        default_table="games",
    )


def _game_country_compare(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    country_attribute = _answer_attribute(context, template_id, "countries")
    game = _game_from_seed(context, seed)
    home_club = _club_by_id(context, game["home_club_id"])
    away_club = _club_by_id(context, game["away_club_id"])
    home_country = _country_by_name(context, home_club["country_name"])
    away_country = _country_by_name(context, away_club["country_name"])
    return _record(
        context,
        template_id,
        {"games": game},
        [
            ("games", game),
            ("clubs", home_club),
            ("clubs", away_club),
            ("countries", home_country),
            ("countries", away_country),
        ],
        _yes_no(home_country[country_attribute] == away_country[country_attribute]),
        "countries",
        home_country,
        country_attribute,
        default_table="games",
    )


def _game_competition_country_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    country_attribute = _answer_attribute(context, template_id, "countries")
    game = _game_from_seed(context, seed)
    competition = _competition_by_id(context, game["competition_id"])
    country = _country_for_competition(context, competition)
    return _record(
        context,
        template_id,
        {"games": game, "competitions": competition, "countries": country},
        [("games", game), ("competitions", competition), ("countries", country)],
        country[country_attribute],
        "countries",
        country,
        country_attribute,
        default_table="games",
    )


def _game_competition_country_compare(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    country_attribute = _answer_attribute(context, template_id, "countries")
    game = _game_from_seed(context, seed)
    competition = _competition_by_id(context, game["competition_id"])
    actual_country = _country_for_competition(context, competition)
    if template_id == 65:
        return _record(
            context,
            template_id,
            {"games": game, "competitions": competition, "countries": actual_country},
            [("games", game), ("competitions", competition), ("countries", actual_country)],
            _yes_no(actual_country[country_attribute] == "Europe"),
            "countries",
            actual_country,
            country_attribute,
            default_table="games",
        )
    comparison_country = _comparison_country_from_seed(context, seed, actual_country)
    return _record(
        context,
        template_id,
        {"games": game, "competitions": competition, "countries": comparison_country},
        [("games", game), ("competitions", competition), ("countries", actual_country), ("countries", comparison_country)],
        _yes_no(actual_country[country_attribute] == comparison_country[country_attribute]),
        "countries",
        actual_country,
        country_attribute,
        default_table="countries",
    )


def _event_club_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    club_attribute = _answer_attribute(context, template_id, "clubs")
    event = _event_from_seed(context, seed)
    club = _club_by_id(context, event["club_id"])
    return _record(
        context,
        template_id,
        {"game_events": event, "clubs": club},
        [("game_events", event), ("clubs", club)],
        club[club_attribute],
        "clubs",
        club,
        club_attribute,
        default_table="game_events",
    )


def _event_club_country_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    country_attribute = _answer_attribute(context, template_id, "countries")
    event = _event_from_seed(context, seed)
    club = _club_by_id(context, event["club_id"])
    country = _country_by_name(context, club["country_name"])
    return _record(
        context,
        template_id,
        {"game_events": event, "clubs": club, "countries": country},
        [("game_events", event), ("clubs", club), ("countries", country)],
        country[country_attribute],
        "countries",
        country,
        country_attribute,
        default_table="game_events",
    )


def _event_player_field(
    context: Level2Context,
    template_id: int,
    seed: dict[str, str],
) -> dict[str, object]:
    player_attribute = _answer_attribute(context, template_id, "players")
    event = _event_from_seed(context, seed)
    player = _player_by_id(context, event["player_id"])
    return _record(
        context,
        template_id,
        {"game_events": event, "players": player},
        [("game_events", event), ("players", player)],
        player[player_attribute],
        "players",
        player,
        player_attribute,
        default_table="game_events",
    )

def question_0(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 0."""
    return _player_country_field(context, 0, seed)

def question_1(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 1."""
    return _player_country_field(context, 1, seed)

def question_2(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 2."""
    return _player_country_field(context, 2, seed)

def question_3(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 3."""
    return _player_country_field(context, 3, seed)

def question_4(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 4."""
    return _player_country_field(context, 4, seed)

def question_5(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 5."""
    return _player_country_field(context, 5, seed)

def question_6(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 6."""
    return _player_country_compare(context, 6, seed)

def question_7(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 7."""
    return _player_country_compare(context, 7, seed)

def question_8(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 8."""
    return _player_country_compare(context, 8, seed)

def question_9(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 9."""
    return _club_country_field(context, 9, seed)

def question_10(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 10."""
    return _club_country_field(context, 10, seed)

def question_11(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 11."""
    return _club_country_field(context, 11, seed)

def question_12(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 12."""
    return _club_country_field(context, 12, seed)

def question_13(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 13."""
    return _club_country_field(context, 13, seed)

def question_14(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 14."""
    return _club_country_field(context, 14, seed)

def question_15(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 15."""
    return _club_competition_field(context, 15, seed)

def question_16(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 16."""
    return _club_competition_field(context, 16, seed)

def question_17(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 17."""
    return _club_competition_field(context, 17, seed)

def question_18(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 18."""
    return _club_competition_field(context, 18, seed)

def question_19(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 19."""
    return _club_competition_field(context, 19, seed)

def question_20(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 20."""
    return _club_competition_field(context, 20, seed)

def question_21(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 21."""
    return _club_competition_country_field(context, 21, seed)

def question_22(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 22."""
    return _club_competition_country_field(context, 22, seed)

def question_23(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 23."""
    return _club_competition_country_field(context, 23, seed)

def question_24(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 24."""
    return _club_competition_country_compare(context, 24, seed)

def question_25(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 25."""
    return _club_competition_country_compare(context, 25, seed)

def question_26(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 26."""
    return _club_competition_country_compare(context, 26, seed)

def question_27(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 27."""
    return _appearance_player_field(context, 27, seed)

def question_28(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 28."""
    return _appearance_player_field(context, 28, seed)

def question_29(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 29."""
    return _appearance_player_field(context, 29, seed)

def question_30(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 30."""
    return _appearance_player_field(context, 30, seed)

def question_31(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 31."""
    return _appearance_player_field(context, 31, seed)

def question_32(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 32."""
    return _appearance_player_field(context, 32, seed)

def question_33(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 33."""
    return _appearance_player_age(context, 33, seed)

def question_34(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 34."""
    return _appearance_player_country_field(context, 34, seed)

def question_35(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 35."""
    return _appearance_player_country_field(context, 35, seed)

def question_36(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 36."""
    return _appearance_player_country_field(context, 36, seed)

def question_37(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 37."""
    return _appearance_player_country_field(context, 37, seed)

def question_38(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 38."""
    return _appearance_player_country_field(context, 38, seed)

def question_39(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 39."""
    return _appearance_player_country_field(context, 39, seed)

def question_40(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 40."""
    return _appearance_club_field(context, 40, seed)

def question_41(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 41."""
    return _appearance_club_field(context, 41, seed)

def question_42(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 42."""
    return _appearance_club_field(context, 42, seed)

def question_43(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 43."""
    return _appearance_club_field(context, 43, seed)

def question_44(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 44."""
    return _appearance_club_field(context, 44, seed)

def question_45(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 45."""
    return _appearance_club_field(context, 45, seed)

def question_46(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 46."""
    return _game_club_field(context, 46, seed)

def question_47(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 47."""
    return _game_club_field(context, 47, seed)

def question_48(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 48."""
    return _game_club_field(context, 48, seed)

def question_49(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 49."""
    return _game_club_field(context, 49, seed)

def question_50(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 50."""
    return _game_club_field(context, 50, seed)

def question_51(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 51."""
    return _game_club_field(context, 51, seed)

def question_52(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 52."""
    return _game_club_compare(context, 52, seed)

def question_53(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 53."""
    return _game_club_compare(context, 53, seed)

def question_54(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 54."""
    return _game_club_country_field(context, 54, seed)

def question_55(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 55."""
    return _game_club_country_field(context, 55, seed)

def question_56(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 56."""
    return _game_club_country_field(context, 56, seed)

def question_57(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 57."""
    return _game_club_country_field(context, 57, seed)

def question_58(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 58."""
    return _game_club_country_field(context, 58, seed)

def question_59(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 59."""
    return _game_club_country_field(context, 59, seed)

def question_60(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 60."""
    return _game_country_compare(context, 60, seed)

def question_61(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 61."""
    return _game_country_compare(context, 61, seed)

def question_62(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 62."""
    return _game_competition_country_field(context, 62, seed)

def question_63(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 63."""
    return _game_competition_country_field(context, 63, seed)

def question_64(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 64."""
    return _game_competition_country_field(context, 64, seed)

def question_65(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 65."""
    return _game_competition_country_compare(context, 65, seed)

def question_66(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 66."""
    return _game_competition_country_compare(context, 66, seed)

def question_67(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 67."""
    return _event_club_field(context, 67, seed)

def question_68(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 68."""
    return _event_club_field(context, 68, seed)

def question_69(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 69."""
    return _event_club_field(context, 69, seed)

def question_70(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 70."""
    return _event_club_country_field(context, 70, seed)

def question_71(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 71."""
    return _event_club_country_field(context, 71, seed)

def question_72(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 72."""
    return _event_club_country_field(context, 72, seed)

def question_73(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 73."""
    return _event_player_field(context, 73, seed)

def question_74(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 74."""
    return _event_player_field(context, 74, seed)

def question_75(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 75."""
    return _event_player_field(context, 75, seed)

def question_76(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 76."""
    return _event_player_field(context, 76, seed)

def question_77(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 77."""
    return _event_player_field(context, 77, seed)

def question_78(context: Level2Context, seed: dict[str, str]) -> dict[str, object]:
    """Generate level 2 template question 78."""
    return _event_player_field(context, 78, seed)

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
    61: question_61,
    62: question_62,
    63: question_63,
    64: question_64,
    65: question_65,
    66: question_66,
    67: question_67,
    68: question_68,
    69: question_69,
    70: question_70,
    71: question_71,
    72: question_72,
    73: question_73,
    74: question_74,
    75: question_75,
    76: question_76,
    77: question_77,
    78: question_78,
}


def generate_level_2_questions(
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    seed_path: Path = DEFAULT_SEED_PATH,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> list[dict[str, object]]:
    """Generate all level 2 questions from seed rows."""
    context = Level2Context.load(template_path=template_path, dataset_dir=dataset_dir)
    seeds_by_table = parse_seed_file(seed_path)

    records: list[dict[str, object]] = []
    for template_id in sorted(context.templates):
        template = context.templates[template_id]
        tables_key = _normalize_tables_key(template["tables"])
        if tables_key not in seeds_by_table:
            continue
        if template_id not in QUESTION_FUNCTIONS:
            raise KeyError(f"No level 2 generator function for template {template_id}")
        for seed in seeds_by_table[tables_key]:
            record = QUESTION_FUNCTIONS[template_id](context, seed)
            records.append({"question_id": len(records), **record})
    return records


def write_questions_json(records: list[dict[str, object]], output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Write generated benchmark records as a JSON array."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def generate_and_write_level_2(
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    seed_path: Path = DEFAULT_SEED_PATH,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, object]]:
    """Generate level 2 records and write them to disk."""
    records = generate_level_2_questions(
        template_path=template_path,
        seed_path=seed_path,
        dataset_dir=dataset_dir,
    )
    write_questions_json(records, output_path=output_path)
    return records
