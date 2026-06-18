"""Generate level 3 benchmark questions with answers and provenance."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATE_PATH = Path("question_template/level_3.csv")
DEFAULT_SEED_PATH = Path("question_template/seed/level_3.txt")
DEFAULT_DATASET_DIR = Path("dataset_clean")
DEFAULT_OUTPUT_PATH = Path("questions/level_3")

TABLE_ID_COLUMNS = {
    "players": "player_id",
    "countries": "country_id",
    "clubs": "club_id",
    "competitions": "competition_id",
    "appearance": "appearance_id",
    "game_events": "game_event_id",
}

TABLE_FILES = {
    "players": "players.csv",
    "countries": "countries.csv",
    "clubs": "clubs.csv",
    "competitions": "competitions.csv",
    "appearance": "appearances.csv",
    "game_events": "game_events.csv",
    "transfers": "transfers.csv",
}

ATTRIBUTE_ALIASES = {
    ("appearance", "competitions_id"): "competition_id",
    ("clubs", "domestic_competitions_id"): "domestic_competition_id",
    ("competitions", "competitions_id"): "competition_id",
}

PLACEHOLDER_RE = re.compile(r"{(?P<name>[^{}]+)}")
OPEN_END = date(9999, 12, 31)


@dataclass(frozen=True)
class Stint:
    player_id: str
    player_name: str
    club_id: str
    club_name: str
    start: date
    end: date | None
    start_transfer: dict[str, str]
    end_transfer: dict[str, str] | None


@dataclass(frozen=True)
class Level3Context:
    templates: dict[int, dict[str, str]]
    table_rows: dict[str, dict[str, dict[str, str]]]
    table_lists: dict[str, list[dict[str, str]]]
    rows_by_name: dict[str, dict[str, dict[str, str]]]
    transfers_by_player_id: dict[str, list[dict[str, str]]]
    transfers_by_player_name: dict[str, list[dict[str, str]]]
    appearances_by_player_id: dict[str, list[dict[str, str]]]
    game_events_by_game_id: dict[str, list[dict[str, str]]]
    stints_by_club_id: dict[str, list[Stint]]

    @classmethod
    def load(
        cls,
        template_path: Path = DEFAULT_TEMPLATE_PATH,
        dataset_dir: Path = DEFAULT_DATASET_DIR,
    ) -> "Level3Context":
        templates = {int(row["question_id"]): row for row in _read_csv_rows(template_path)}
        table_lists = {
            table: _read_csv_rows(dataset_dir / filename)
            for table, filename in TABLE_FILES.items()
        }
        table_rows = {
            table: _index_rows_by_id(rows, table)
            for table, rows in table_lists.items()
            if table in TABLE_ID_COLUMNS
        }
        rows_by_name = {
            "countries": {
                row["country_name"]: row
                for row in table_rows["countries"].values()
                if row.get("country_name")
            },
            "clubs": {
                row["name"]: row
                for row in table_rows["clubs"].values()
                if row.get("name")
            },
        }

        transfers_by_player_id: dict[str, list[dict[str, str]]] = defaultdict(list)
        transfers_by_player_name: dict[str, list[dict[str, str]]] = defaultdict(list)
        for transfer in table_lists["transfers"]:
            transfers_by_player_id[transfer["player_id"]].append(transfer)
            transfers_by_player_name[transfer["player_name"]].append(transfer)
        for rows in transfers_by_player_id.values():
            rows.sort(key=_transfer_sort_key)
        for rows in transfers_by_player_name.values():
            rows.sort(key=_transfer_sort_key)

        appearances_by_player_id: dict[str, list[dict[str, str]]] = defaultdict(list)
        for appearance in table_lists["appearance"]:
            appearances_by_player_id[appearance["player_id"]].append(appearance)
        for rows in appearances_by_player_id.values():
            rows.sort(key=lambda row: (row["date"], row["appearance_id"]))

        game_events_by_game_id: dict[str, list[dict[str, str]]] = defaultdict(list)
        for event in table_lists["game_events"]:
            game_events_by_game_id[event["game_id"]].append(event)
        for rows in game_events_by_game_id.values():
            rows.sort(key=lambda row: (_safe_int(row.get("minute", "")), row["game_event_id"]))

        stints_by_club_id = _build_stints_by_club_id(transfers_by_player_id)
        return cls(
            templates=templates,
            table_rows=table_rows,
            table_lists=table_lists,
            rows_by_name=rows_by_name,
            transfers_by_player_id=dict(transfers_by_player_id),
            transfers_by_player_name=dict(transfers_by_player_name),
            appearances_by_player_id=dict(appearances_by_player_id),
            game_events_by_game_id=dict(game_events_by_game_id),
            stints_by_club_id=stints_by_club_id,
        )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _index_rows_by_id(rows: list[dict[str, str]], table: str) -> dict[str, dict[str, str]]:
    id_column = TABLE_ID_COLUMNS[table]
    indexed = {}
    for row in rows:
        row_id = row.get(id_column, "")
        if not row_id:
            raise ValueError(f"{table}: row is missing {id_column}: {row}")
        indexed[str(row_id)] = row
    return indexed


def _transfer_sort_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("transfer_date", ""),
        row.get("player_id", ""),
        row.get("from_club_id", ""),
        row.get("to_club_id", ""),
    )


def _safe_int(value: str | int | None, default: int = 0) -> int:
    try:
        return int(value or default)
    except ValueError:
        return default


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _split_csv_line(line: str) -> list[str]:
    return next(csv.reader([line], skipinitialspace=True))


def _strip_optional_braces(value: str) -> str:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        return value[1:-1].strip()
    return value


def _parse_key_value_row(line: str) -> dict[str, str]:
    values = {}
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
    if not path.exists():
        raise FileNotFoundError(f"Missing seed file: {path}")
    grouped_seeds: dict[str, list[dict[str, str]]] = defaultdict(list)
    current_tables = None
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


def _normalize_attribute(table: str, attribute: str) -> str:
    return ATTRIBUTE_ALIASES.get((table, attribute), attribute)


def _json_scalar(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _row_identity(table: str, row: dict[str, str]) -> dict[str, int | str]:
    if table == "transfers":
        attributes = ["player_id", "transfer_date", "from_club_id", "to_club_id", "player_name"]
        return {
            "id_attribute": "+".join(attributes),
            "id_value": "|".join(row.get(attribute, "") for attribute in attributes),
        }
    id_attribute = TABLE_ID_COLUMNS[table]
    return {"id_attribute": id_attribute, "id_value": _json_scalar(row[id_attribute])}


def _row_ref(table: str, row: dict[str, str]) -> dict[str, int | str]:
    return {"table": table, **_row_identity(table, row)}


def _dedupe_row_refs(rows: list[tuple[str, dict[str, str] | None]]) -> list[dict[str, int | str]]:
    refs = []
    seen = set()
    for table, row in rows:
        if row is None:
            continue
        ref = _row_ref(table, row)
        key = (ref["table"], ref["id_attribute"], ref["id_value"])
        if key not in seen:
            seen.add(key)
            refs.append(ref)
    return refs


def _derived_source(operation: str, attributes: list[str], description: str) -> dict[str, Any]:
    return {
        "type": "derived",
        "operation": operation,
        "attributes": attributes,
        "description": description,
    }


def _cell_source(table: str, row: dict[str, str], attribute: str) -> dict[str, Any]:
    return {"table": table, **_row_identity(table, row), "attribute": attribute}


def _placeholder_names(template: str) -> list[str]:
    seen = set()
    names = []
    for match in PLACEHOLDER_RE.finditer(template):
        name = match.group("name").strip()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _table_value(context: Level3Context, table: str, row: dict[str, str], attribute: str) -> str:
    attribute = _normalize_attribute(table, attribute)
    if attribute in row:
        return row[attribute]
    if table == "clubs" and attribute == "country_name":
        return _club_country_name(context, row)
    raise KeyError(f"{table} row has no column {attribute!r}")


def _fill_template(
    context: Level3Context,
    template: str,
    fill_rows: dict[str, dict[str, str]],
    extra_values: dict[str, str],
) -> tuple[str, dict[str, str]]:
    fill_values = {}
    placeholders = {}
    for placeholder in _placeholder_names(template):
        if placeholder in extra_values:
            fill_values[placeholder] = extra_values[placeholder]
            placeholders[placeholder] = extra_values[placeholder]
            continue
        if "." in placeholder:
            table, attribute = placeholder.split(".", 1)
            attribute = _normalize_attribute(table, attribute)
            if table not in fill_rows:
                raise KeyError(f"No row available for placeholder table {table!r}")
            value = _table_value(context, table, fill_rows[table], attribute)
            fill_values[placeholder] = value
            placeholders[f"{table}.{attribute}"] = value
            continue
        candidates = [
            table
            for table, row in fill_rows.items()
            if _normalize_attribute(table, placeholder) in row
        ]
        if len(candidates) != 1:
            raise KeyError(
                f"Could not resolve unqualified placeholder {placeholder!r}; "
                f"candidate tables: {candidates}"
            )
        table = candidates[0]
        attribute = _normalize_attribute(table, placeholder)
        value = _table_value(context, table, fill_rows[table], attribute)
        fill_values[placeholder] = value
        placeholders[f"{table}.{attribute}"] = value
    return PLACEHOLDER_RE.sub(lambda match: fill_values[match.group("name").strip()], template), placeholders


def _record(
    context: Level3Context,
    template_id: int,
    fill_rows: dict[str, dict[str, str]],
    extra_values: dict[str, str],
    ground_truth_rows: list[tuple[str, dict[str, str] | None]],
    answer: str | int,
    answer_source: dict[str, Any],
) -> dict[str, Any]:
    question, placeholders = _fill_template(
        context,
        context.templates[template_id]["question"],
        fill_rows,
        extra_values,
    )
    return {
        "template_question_id": template_id,
        "question": question,
        "answer": str(answer),
        "ground_truth": {
            "rows": _dedupe_row_refs(ground_truth_rows),
            "placeholders": placeholders,
            "answer_source": answer_source,
        },
    }


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _format_list(values: list[str]) -> str:
    return "; ".join(values) if values else "NULL"


def _mode_values(values: list[str]) -> list[str]:
    values = [value for value in values if value]
    if not values:
        return []
    counts = Counter(values)
    max_count = max(counts.values())
    modes = []
    seen = set()
    for value in values:
        if counts[value] == max_count and value not in seen:
            seen.add(value)
            modes.append(value)
    return modes


def _age_on(player: dict[str, str], target: date) -> int:
    born = _parse_date(player["date_of_birth"])
    return target.year - born.year - ((target.month, target.day) < (born.month, born.day))


def _build_stints_by_club_id(
    transfers_by_player_id: dict[str, list[dict[str, str]]],
) -> dict[str, list[Stint]]:
    stints_by_club_id: dict[str, list[Stint]] = defaultdict(list)
    for player_id, transfers in transfers_by_player_id.items():
        sorted_transfers = sorted(transfers, key=_transfer_sort_key)
        for index, transfer in enumerate(sorted_transfers):
            club_id = transfer.get("to_club_id", "")
            if not club_id:
                continue
            end_transfer = sorted_transfers[index + 1] if index + 1 < len(sorted_transfers) else None
            stint = Stint(
                player_id=player_id,
                player_name=transfer["player_name"],
                club_id=club_id,
                club_name=transfer["to_club_name"],
                start=_parse_date(transfer["transfer_date"]),
                end=_parse_date(end_transfer["transfer_date"]) if end_transfer else None,
                start_transfer=transfer,
                end_transfer=end_transfer,
            )
            stints_by_club_id[club_id].append(stint)
    for stints in stints_by_club_id.values():
        stints.sort(key=lambda stint: (stint.start, stint.player_name, stint.player_id))
    return dict(stints_by_club_id)


def _seed_get(seed: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        if key in seed and seed[key] != "":
            return seed[key]
    return None


def _country_by_name(context: Level3Context, country_name: str) -> dict[str, str]:
    row = context.rows_by_name["countries"].get(country_name)
    if row is None:
        raise ValueError(f"No country row found for country_name={country_name!r}")
    return row


def _maybe_country_by_name(context: Level3Context, country_name: str) -> dict[str, str] | None:
    if not country_name:
        return None
    return context.rows_by_name["countries"].get(country_name)


def _country_by_seed_or_fallback(
    context: Level3Context,
    seed: dict[str, str],
    fallback: dict[str, str],
    attribute_hint: str | None = None,
) -> dict[str, str]:
    if attribute_hint and attribute_hint != "country_name":
        value = _seed_get(seed, f"countries.{attribute_hint}", attribute_hint)
        if value:
            for row in context.table_rows["countries"].values():
                if row.get(attribute_hint) == value:
                    return row
            raise ValueError(f"No country row found for {attribute_hint}={value!r}")
    country_id = _seed_get(seed, "countries.country_id", "country_id")
    if country_id:
        row = context.table_rows["countries"].get(str(country_id))
        if row is None:
            raise ValueError(f"No countries row found for country_id={country_id!r}")
        return row
    country_name = _seed_get(seed, "countries.country_name", "country_name")
    if country_name:
        return _country_by_name(context, country_name)
    return fallback


def _club_by_id(context: Level3Context, club_id: str) -> dict[str, str]:
    row = context.table_rows["clubs"].get(str(club_id))
    if row is None:
        raise ValueError(f"No club row found for club_id={club_id!r}")
    return row


def _maybe_club_by_id(context: Level3Context, club_id: str) -> dict[str, str] | None:
    return context.table_rows["clubs"].get(str(club_id))


def _player_by_id(context: Level3Context, player_id: str) -> dict[str, str]:
    row = context.table_rows["players"].get(str(player_id))
    if row is None:
        raise ValueError(f"No player row found for player_id={player_id!r}")
    return row


def _maybe_player_by_id(context: Level3Context, player_id: str) -> dict[str, str] | None:
    return context.table_rows["players"].get(str(player_id))


def _competition_by_id(context: Level3Context, competition_id: str) -> dict[str, str]:
    row = context.table_rows["competitions"].get(str(competition_id))
    if row is None:
        raise ValueError(f"No competition row found for competition_id={competition_id!r}")
    return row


def _maybe_competition_by_id(context: Level3Context, competition_id: str) -> dict[str, str] | None:
    return context.table_rows["competitions"].get(str(competition_id))


def _maybe_competition_for_club(context: Level3Context, club: dict[str, str]) -> dict[str, str] | None:
    return _maybe_competition_by_id(context, club.get("domestic_competition_id", ""))


def _competition_for_club(context: Level3Context, club: dict[str, str]) -> dict[str, str]:
    return _competition_by_id(context, club.get("domestic_competition_id", ""))


def _country_for_competition(context: Level3Context, competition: dict[str, str]) -> dict[str, str]:
    country_id = competition.get("country_id", "")
    if country_id and country_id != "-1":
        row = context.table_rows["countries"].get(country_id)
        if row is not None:
            return row
    if competition.get("country_name"):
        return _country_by_name(context, competition["country_name"])
    raise ValueError(f"Could not resolve country for competition {competition['competition_id']!r}")


def _club_country_name(context: Level3Context, club: dict[str, str]) -> str:
    if club.get("country_name"):
        return club["country_name"]
    competition = _maybe_competition_for_club(context, club)
    if competition and competition.get("country_name"):
        return competition["country_name"]
    raise ValueError(f"Could not resolve country for club {club.get('club_id', '')!r}")


def _club_country(context: Level3Context, club: dict[str, str]) -> dict[str, str]:
    return _country_by_name(context, _club_country_name(context, club))


def _transfer_rows_for_seed(
    context: Level3Context,
    seed: dict[str, str],
    apply_specific_filters: bool = True,
) -> list[dict[str, str]]:
    player_id = _seed_get(seed, "transfers.player_id", "player_id")
    if player_id:
        transfers = list(context.transfers_by_player_id.get(player_id, []))
    else:
        player_name = _seed_get(seed, "transfers.player_name", "player_name")
        transfers = list(context.transfers_by_player_name.get(player_name, [])) if player_name else list(context.table_lists["transfers"])
    if apply_specific_filters or not (player_id or player_name):
        filters = {
            "transfer_date": _seed_get(seed, "transfers.transfer_date", "transfer_date"),
            "transfer_season": _seed_get(seed, "transfers.transfer_season", "transfer_season"),
            "from_club_id": _seed_get(seed, "transfers.from_club_id", "from_club_id"),
            "to_club_id": _seed_get(seed, "transfers.to_club_id", "to_club_id"),
        }
        for attribute, value in filters.items():
            if value:
                transfers = [row for row in transfers if row.get(attribute) == value]
    transfers.sort(key=_transfer_sort_key)
    if not transfers:
        raise ValueError(f"Seed {seed!r} did not match any transfer rows")
    return transfers


def _specific_transfer(context: Level3Context, seed: dict[str, str]) -> dict[str, str]:
    transfers = _transfer_rows_for_seed(context, seed)
    for transfer in transfers:
        if _maybe_club_by_id(context, transfer.get("from_club_id", "")) and _maybe_club_by_id(context, transfer.get("to_club_id", "")):
            return transfer
    return transfers[0]


def _next_transfer(context: Level3Context, transfer: dict[str, str]) -> dict[str, str] | None:
    transfers = context.transfers_by_player_id.get(transfer["player_id"], [])
    key = _transfer_sort_key(transfer)
    for index, candidate in enumerate(transfers):
        if _transfer_sort_key(candidate) == key:
            return transfers[index + 1] if index + 1 < len(transfers) else None
    transfer_date = _parse_date(transfer["transfer_date"])
    for candidate in transfers:
        if _parse_date(candidate["transfer_date"]) > transfer_date:
            return candidate
    return None


def _previous_transfer(context: Level3Context, transfer: dict[str, str]) -> dict[str, str] | None:
    transfers = context.transfers_by_player_id.get(transfer["player_id"], [])
    key = _transfer_sort_key(transfer)
    for index, candidate in enumerate(transfers):
        if _transfer_sort_key(candidate) == key:
            return transfers[index - 1] if index > 0 else None
    transfer_date = _parse_date(transfer["transfer_date"])
    previous = [candidate for candidate in transfers if _parse_date(candidate["transfer_date"]) < transfer_date]
    return previous[-1] if previous else None


def _target_date(seed: dict[str, str], transfer: dict[str, str]) -> date:
    value = _seed_get(seed, "target_date")
    return _parse_date(value) if value else _parse_date(transfer["transfer_date"])


def _current_transfer_at(context: Level3Context, seed: dict[str, str]) -> dict[str, str]:
    transfers = _transfer_rows_for_seed(context, seed, apply_specific_filters=False)
    target = _target_date(seed, transfers[0])
    candidates = [
        transfer
        for transfer in transfers
        if _parse_date(transfer["transfer_date"]) <= target
        and _maybe_club_by_id(context, transfer.get("to_club_id", "")) is not None
    ]
    if not candidates:
        raise ValueError(
            f"Cannot determine current club on {target.isoformat()} because no joined transfer "
            "exists on or before that date"
        )
    return candidates[-1]


def _event_from_seed(context: Level3Context, seed: dict[str, str]) -> dict[str, str]:
    event_id = _seed_get(seed, "game_events.game_event_id", "game_event_id")
    if event_id:
        row = context.table_rows["game_events"].get(event_id)
        if row is not None:
            return row
        events_for_game = context.game_events_by_game_id.get(event_id, [])
        if events_for_game:
            return events_for_game[0]
        raise ValueError(f"No game_events row found for game_event_id/game_id={event_id!r}")
    game_id = _seed_get(seed, "game_events.game_id", "game_id")
    if game_id:
        events_for_game = context.game_events_by_game_id.get(game_id, [])
        if events_for_game:
            return events_for_game[0]
    raise ValueError(f"Seed {seed!r} needs game_events.game_event_id or game_events.game_id")


def _destination_country_items(context: Level3Context, transfers: list[dict[str, str]], country_attribute: str):
    items = []
    missing_club_ids = []
    for transfer in transfers:
        club = _maybe_club_by_id(context, transfer.get("to_club_id", ""))
        if club is None:
            missing_club_ids.append(transfer.get("to_club_id", ""))
            continue
        if country_attribute == "country_name":
            items.append((transfer, club, None, _club_country_name(context, club)))
        else:
            country = _club_country(context, club)
            items.append((transfer, club, country, country[country_attribute]))
    if missing_club_ids:
        raise ValueError(
            "Cannot answer destination-club aggregate because these to_club_id values "
            f"are missing from clubs.csv: {sorted(set(missing_club_ids))}"
        )
    return items


def _transfer_country_pair_items(context: Level3Context, transfers: list[dict[str, str]], country_attribute: str):
    items = []
    missing_club_ids = []
    for transfer in transfers:
        from_club = _maybe_club_by_id(context, transfer.get("from_club_id", ""))
        to_club = _maybe_club_by_id(context, transfer.get("to_club_id", ""))
        if from_club is None:
            missing_club_ids.append(transfer.get("from_club_id", ""))
        if to_club is None:
            missing_club_ids.append(transfer.get("to_club_id", ""))
        if from_club is None or to_club is None:
            continue
        if country_attribute == "country_name":
            items.append((transfer, from_club, to_club, None, None, _club_country_name(context, from_club), _club_country_name(context, to_club)))
        else:
            from_country = _club_country(context, from_club)
            to_country = _club_country(context, to_club)
            items.append((transfer, from_club, to_club, from_country, to_country, from_country[country_attribute], to_country[country_attribute]))
    if missing_club_ids:
        raise ValueError(
            "Cannot answer from/to-club transfer aggregate because these club ids "
            f"are missing from clubs.csv: {sorted(set(missing_club_ids))}"
        )
    return items


def _destination_ground_rows(items: list[tuple[Any, ...]]) -> list[tuple[str, dict[str, str] | None]]:
    rows = []
    for transfer, club, country, _value in items:
        rows.extend([("transfers", transfer), ("clubs", club), ("countries", country)])
    return rows


def _pair_ground_rows(items: list[tuple[Any, ...]]) -> list[tuple[str, dict[str, str] | None]]:
    rows = []
    for transfer, from_club, to_club, from_country, to_country, _from_value, _to_value in items:
        rows.extend([("transfers", transfer), ("clubs", from_club), ("clubs", to_club), ("countries", from_country), ("countries", to_country)])
    return rows


def _transfer_aggregate_question(context: Level3Context, template_id: int, seed: dict[str, str], country_attribute: str) -> dict[str, Any]:
    transfers = _transfer_rows_for_seed(context, seed, apply_specific_filters=False)
    items = _destination_country_items(context, transfers, country_attribute)
    if not items:
        raise ValueError(f"Template {template_id}: no destination club joins for seed {seed!r}")
    values = [item[3] for item in items]
    fill_rows = {"transfers": items[0][0], "clubs": items[0][1]}
    ground_rows = _destination_ground_rows(items)
    extra_values = {}
    operation = "derived"

    if template_id in {0, 12, 21}:
        answer = len(set(values))
        operation = "count_distinct"
    elif template_id in {1, 13, 22}:
        answer = _format_list(_mode_values(values))
        operation = "mode"
    elif template_id in {2, 23}:
        fallback_country = items[0][2] or _club_country(context, items[0][1])
        hint = "country_name" if template_id == 2 else country_attribute
        country = _country_by_seed_or_fallback(context, seed, fallback_country, hint)
        fill_rows["countries"] = country
        ground_rows.append(("countries", country))
        answer = sum(1 for value in values if value == country[hint])
        operation = "count_matches_placeholder"
    elif template_id == 14:
        answer = sum(1 for value in values if value == "Europe")
        operation = "count_matches_fixed_europe"
    elif template_id in {3, 15}:
        answer = _format_list(values)
        operation = "list_in_transfer_date_order"
    elif template_id in {4, 16}:
        answer = values[0]
        operation = "first_by_transfer_date"
    elif template_id in {5, 17}:
        answer = values[-1]
        operation = "latest_by_transfer_date"
    elif template_id == 10:
        season = _seed_get(seed, "transfer_season", "transfers.transfer_season") or items[0][0]["transfer_season"]
        extra_values["transfer_season"] = season
        answer = len({value for transfer, _club, _country, value in items if transfer["transfer_season"] == season})
        operation = "count_distinct_in_season"
    elif template_id == 11:
        by_season: dict[str, set[str]] = defaultdict(set)
        for transfer, _club, _country, value in items:
            by_season[transfer["transfer_season"]].add(value)
        max_count = max(len(country_names) for country_names in by_season.values())
        answer = _format_list([season for season, country_names in by_season.items() if len(country_names) == max_count])
        operation = "season_with_max_distinct"
    elif template_id == 24:
        answer = _yes_no(len(set(values)) <= 1)
        operation = "all_same"
    else:
        raise KeyError(f"Template {template_id}: unsupported transfer aggregate")

    return _record(
        context,
        template_id,
        fill_rows,
        extra_values,
        ground_rows,
        answer,
        _derived_source(operation, ["transfers.to_club_id", "clubs.country_name" if country_attribute == "country_name" else f"countries.{country_attribute}"], "Computed from the seed player's destination club joins."),
    )


def _transfer_pair_question(context: Level3Context, template_id: int, seed: dict[str, str], country_attribute: str) -> dict[str, Any]:
    transfers = _transfer_rows_for_seed(context, seed, apply_specific_filters=False)
    items = _transfer_country_pair_items(context, transfers, country_attribute)
    if not items:
        raise ValueError(f"Template {template_id}: no from/to club joins for seed {seed!r}")
    fill_rows = {"transfers": items[0][0], "clubs": items[0][2]}
    ground_rows = _pair_ground_rows(items)
    if template_id in {6, 18}:
        answer = sum(1 for item in items if item[5] == item[6])
        operation = "count_same"
    elif template_id in {7, 19}:
        answer = sum(1 for item in items if item[5] != item[6])
        operation = "count_different"
    elif template_id in {8, 20}:
        answer = _yes_no(any(item[5] != item[6] for item in items))
        operation = "exists_different"
    elif template_id == 9:
        fallback_country = _club_country(context, items[0][1])
        country = _country_by_seed_or_fallback(context, seed, fallback_country, "country_name")
        fill_rows["countries"] = country
        ground_rows.append(("countries", country))
        answer = sum(1 for item in items if item[5] == country["country_name"] and item[6] != country["country_name"])
        operation = "count_from_placeholder_country_to_different_country"
    else:
        raise KeyError(f"Template {template_id}: unsupported transfer pair aggregate")
    return _record(
        context,
        template_id,
        fill_rows,
        {},
        ground_rows,
        answer,
        _derived_source(operation, ["transfers.from_club_id", "transfers.to_club_id", "clubs.country_name" if country_attribute == "country_name" else f"countries.{country_attribute}"], "Joined each transfer to previous and destination clubs."),
    )


def _event_game_rows(context: Level3Context, seed: dict[str, str]) -> tuple[dict[str, str], list[dict[str, str]]]:
    seed_event = _event_from_seed(context, seed)
    events = context.game_events_by_game_id.get(seed_event["game_id"], [])
    if not events:
        raise ValueError(f"No game events found for game_id={seed_event['game_id']!r}")
    return seed_event, events


def _event_country_fallback(context: Level3Context, event: dict[str, str]) -> dict[str, str]:
    club = _maybe_club_by_id(context, event.get("club_id", ""))
    if club is not None:
        return _club_country(context, club)
    player = _maybe_player_by_id(context, event.get("player_id", ""))
    if player is not None:
        country = _maybe_country_by_name(context, player.get("country_of_citizenship", ""))
        if country is not None:
            return country
    return next(iter(context.table_rows["countries"].values()))


def _game_event_question(context: Level3Context, template_id: int, seed: dict[str, str]) -> dict[str, Any]:
    seed_event, events = _event_game_rows(context, seed)
    fill_rows = {"game_events": seed_event}
    comparison_country: dict[str, str] | None = None
    comparison_continent = "Europe" if template_id in {25, 26, 28, 32, 33} else None
    ground_rows: list[tuple[str, dict[str, str] | None]] = [("game_events", seed_event)]
    if template_id != 31 and comparison_continent is None:
        country_hint = "capital_city" if template_id == 27 else "country_name"
        comparison_country = _country_by_seed_or_fallback(context, seed, _event_country_fallback(context, seed_event), country_hint)
        fill_rows["countries"] = comparison_country
        ground_rows.append(("countries", comparison_country))
    count = 0
    values: list[str] = []

    for event in events:
        ground_rows.append(("game_events", event))
        if template_id in {25, 29, 30, 31, 32, 33}:
            club = _maybe_club_by_id(context, event.get("club_id", ""))
            if club is None:
                continue
            ground_rows.append(("clubs", club))
            country = _club_country(context, club)
            if template_id in {25, 32, 33}:
                ground_rows.append(("countries", country))
            if template_id == 25 and event["type"] == seed_event["type"] and country["continent"] == comparison_continent:
                count += 1
            elif template_id == 29 and _club_country_name(context, club) == comparison_country["country_name"]:
                count += 1
            elif template_id == 30 and event["type"] == seed_event["type"] and _club_country_name(context, club) == comparison_country["country_name"]:
                count += 1
            elif template_id == 31:
                values.append(_club_country_name(context, club))
            elif template_id == 32 and country["continent"] == comparison_continent:
                count += 1
            elif template_id == 33 and event["type"] == seed_event["type"] and country["continent"] == comparison_continent:
                count += 1
        elif template_id in {26, 27, 28}:
            player_id_attribute = "player_assist_id" if template_id == 28 else "player_id"
            player = _maybe_player_by_id(context, event.get(player_id_attribute, ""))
            if player is None:
                continue
            ground_rows.append(("players", player))
            country_name = player["country_of_birth"] if template_id == 27 else player["country_of_citizenship"]
            country = _maybe_country_by_name(context, country_name)
            if country is None:
                continue
            ground_rows.append(("countries", country))
            if template_id == 26 and country["continent"] == comparison_continent:
                count += 1
            elif template_id == 27 and country["capital_city"] == comparison_country["capital_city"]:
                count += 1
            elif template_id == 28 and country["continent"] == comparison_continent:
                count += 1
        else:
            raise KeyError(f"Template {template_id}: unsupported game event question")
    answer = _format_list(_mode_values(values)) if template_id == 31 else count
    operation = "mode" if template_id == 31 else "count_matches_fixed_europe" if comparison_continent else "count_matches"
    return _record(
        context,
        template_id,
        fill_rows,
        {},
        ground_rows,
        answer,
        _derived_source(operation, [part.strip() for part in context.templates[template_id]["attributes"].split(";")], "Computed from all events in the seed event's game."),
    )


def _current_club_bundle(context: Level3Context, seed: dict[str, str]) -> tuple[dict[str, str], dict[str, str], dict[str, str], date]:
    transfer = _current_transfer_at(context, seed)
    club = _club_by_id(context, transfer["to_club_id"])
    country = _club_country(context, club)
    return transfer, club, country, _target_date(seed, transfer)


def _club_by_seed_or_fallback(context: Level3Context, seed: dict[str, str], fallback: dict[str, str]) -> dict[str, str]:
    club_id = _seed_get(seed, "clubs.club_id", "club_id")
    if club_id:
        return _club_by_id(context, club_id)
    club_name = _seed_get(seed, "clubs.name", "club_name")
    if club_name:
        row = context.rows_by_name["clubs"].get(club_name)
        if row is None:
            raise ValueError(f"No club row found for name={club_name!r}")
        return row
    seats = _seed_get(seed, "clubs.stadium_seats", "stadium_seats")
    if seats:
        for row in context.table_rows["clubs"].values():
            if row.get("stadium_seats") == seats:
                return row
        raise ValueError(f"No club row found for stadium_seats={seats!r}")
    return fallback


def _current_club_country_question(context: Level3Context, template_id: int, seed: dict[str, str]) -> dict[str, Any]:
    transfer, club, country, target = _current_club_bundle(context, seed)
    fill_rows = {"transfers": transfer, "clubs": club}
    ground_rows: list[tuple[str, dict[str, str] | None]] = [("transfers", transfer), ("clubs", club)]
    extra_values = {"target_date": target.isoformat()}

    if template_id == 34:
        answer = _club_country_name(context, club)
        source = _cell_source("clubs", club, "country_name")
    elif template_id == 35:
        answer = club["stadium_name"]
        source = _cell_source("clubs", club, "stadium_name")
    elif template_id == 36:
        answer = club["stadium_seats"]
        source = _cell_source("clubs", club, "stadium_seats")
    elif template_id == 37:
        comparison = _country_by_seed_or_fallback(context, seed, country, "country_name")
        fill_rows["countries"] = comparison
        ground_rows.append(("countries", comparison))
        answer = _yes_no(_club_country_name(context, club) == comparison["country_name"])
        source = _derived_source("compare", ["clubs.country_name", "countries.country_name"], "Compared current club country with the placeholder country.")
    elif template_id in {38, 39, 40}:
        threshold_club = _club_by_seed_or_fallback(context, seed, club)
        fill_rows["clubs"] = threshold_club
        ground_rows.append(("clubs", threshold_club))
        current_seats = _safe_int(club["stadium_seats"])
        threshold = _safe_int(threshold_club["stadium_seats"])
        if template_id == 38:
            answer = _yes_no(current_seats > threshold)
            operation = "greater_than"
        elif template_id == 39:
            answer = _yes_no(current_seats < threshold)
            operation = "less_than"
        else:
            answer = _yes_no(current_seats == threshold)
            operation = "equals"
        source = _derived_source(operation, ["clubs.stadium_seats"], "Compared current stadium seats with the placeholder value.")
    elif template_id == 41:
        stadium_name = _seed_get(seed, "stadium_name", "clubs.stadium_name") or club["stadium_name"]
        extra_values["stadium_name"] = stadium_name
        answer = _yes_no(club["stadium_name"] == stadium_name)
        source = _derived_source("compare", ["clubs.stadium_name"], "Compared current club stadium with the placeholder stadium name.")
    elif template_id in {42, 43, 44}:
        fill_rows["countries"] = country
        ground_rows.append(("countries", country))
        attribute = {42: "continent", 43: "capital_city", 44: "confederation"}[template_id]
        answer = country[attribute]
        source = _cell_source("countries", country, attribute)
    elif template_id == 45:
        fill_rows["countries"] = country
        ground_rows.append(("countries", country))
        answer = _yes_no(country["continent"] == "Europe")
        source = _derived_source("compare_fixed_europe", ["countries.continent"], "Compared current club country continent with Europe.")
    elif template_id in {46, 47, 48}:
        fill_rows["countries"] = country
        ground_rows.append(("countries", country))
        attribute = {46: "confederation", 47: "capital_city", 48: "capital_city"}[template_id]
        comparison = _country_by_seed_or_fallback(context, seed, country, attribute)
        fill_rows["countries"] = comparison
        ground_rows.append(("countries", comparison))
        answer = _yes_no(country[attribute] == comparison[attribute])
        source = _derived_source("compare", [f"countries.{attribute}"], "Compared current club country attribute with the placeholder.")
    else:
        raise KeyError(f"Template {template_id}: unsupported current club question")
    return _record(context, template_id, fill_rows, extra_values, ground_rows, answer, source)


def _current_competition_question(context: Level3Context, template_id: int, seed: dict[str, str]) -> dict[str, Any]:
    transfer, club, country, target = _current_club_bundle(context, seed)
    competition = _competition_for_club(context, club)
    fill_rows = {"transfers": transfer, "clubs": club, "competitions": competition}
    ground_rows: list[tuple[str, dict[str, str] | None]] = [("transfers", transfer), ("clubs", club), ("competitions", competition)]
    extra_values = {"target_date": target.isoformat()}
    if template_id in {49, 50, 51, 52}:
        attribute = {49: "type", 50: "sub_type", 51: "country_name", 52: "confederation"}[template_id]
        answer = competition[attribute]
        source = _cell_source("competitions", competition, attribute)
    elif template_id == 53:
        answer = sum(1 for candidate in context.table_rows["clubs"].values() if candidate.get("domestic_competition_id") == competition["competition_id"])
        source = _derived_source("count", ["clubs.domestic_competition_id", "competitions.competition_id"], "Counted clubs in the current club's domestic competition.")
    else:
        raise KeyError(f"Template {template_id}: unsupported competition question")
    return _record(context, template_id, fill_rows, extra_values, ground_rows, answer, source)


def _current_player_country_question(context: Level3Context, template_id: int, seed: dict[str, str]) -> dict[str, Any]:
    transfer, club, club_country, target = _current_club_bundle(context, seed)
    player = _player_by_id(context, transfer["player_id"])
    player_country_attribute = "country_of_citizenship" if template_id in {56, 58, 60, 62} else "country_of_birth"
    player_country = _maybe_country_by_name(context, player[player_country_attribute])
    ground_rows: list[tuple[str, dict[str, str] | None]] = [("transfers", transfer), ("clubs", club), ("players", player)]
    if template_id in {56, 57}:
        answer = _yes_no(_club_country_name(context, club) == player[player_country_attribute])
        attributes = ["clubs.country_name", f"players.{player_country_attribute}"]
    else:
        ground_rows.extend([("countries", club_country), ("countries", player_country)])
        attribute = {58: "continent", 59: "continent", 60: "confederation", 61: "confederation", 62: "capital_city", 63: "capital_city"}[template_id]
        answer = _yes_no(player_country is not None and club_country[attribute] == player_country[attribute])
        attributes = [f"countries.{attribute}", f"players.{player_country_attribute}"]
    return _record(context, template_id, {"transfers": transfer, "clubs": club, "players": player}, {"target_date": target.isoformat()}, ground_rows, answer, _derived_source("compare", attributes, "Compared current club country with the player's country."))


def _specific_transfer_compare_question(context: Level3Context, template_id: int, seed: dict[str, str]) -> dict[str, Any]:
    transfer = _specific_transfer(context, seed)
    from_club = _club_by_id(context, transfer["from_club_id"])
    to_club = _club_by_id(context, transfer["to_club_id"])
    ground_rows: list[tuple[str, dict[str, str] | None]] = [("transfers", transfer), ("clubs", from_club), ("clubs", to_club)]
    if template_id == 64:
        answer = _yes_no(_club_country_name(context, from_club) == _club_country_name(context, to_club))
        attributes = ["clubs.country_name"]
    else:
        attribute = {65: "continent", 66: "confederation", 67: "capital_city"}[template_id]
        from_country = _club_country(context, from_club)
        to_country = _club_country(context, to_club)
        ground_rows.extend([("countries", from_country), ("countries", to_country)])
        answer = _yes_no(from_country[attribute] == to_country[attribute])
        attributes = [f"countries.{attribute}"]
    return _record(context, template_id, {"transfers": transfer, "clubs": to_club}, {}, ground_rows, answer, _derived_source("compare", attributes, "Compared previous and destination clubs for the selected transfer."))


def _transfer_age_question(context: Level3Context, template_id: int, seed: dict[str, str]) -> dict[str, Any]:
    transfers = _transfer_rows_for_seed(context, seed, apply_specific_filters=False)
    transfer = transfers[0] if template_id == 68 else transfers[-1]
    player = _player_by_id(context, transfer["player_id"])
    answer = _age_on(player, _parse_date(transfer["transfer_date"]))
    return _record(context, template_id, {"transfers": transfer, "players": player}, {}, [("transfers", transfer), ("players", player)], answer, _derived_source("age_on_date", ["players.date_of_birth", "transfers.transfer_date"], "Calculated age on the selected transfer date."))


def _appearance_window(context: Level3Context, transfer: dict[str, str]) -> list[dict[str, str]]:
    start = _parse_date(transfer["transfer_date"])
    next_transfer = _next_transfer(context, transfer)
    end = _parse_date(next_transfer["transfer_date"]) if next_transfer else None
    rows = []
    for appearance in context.appearances_by_player_id.get(transfer["player_id"], []):
        appearance_date = _parse_date(appearance["date"])
        if appearance["player_club_id"] != transfer["to_club_id"] or appearance_date < start:
            continue
        if end is not None and appearance_date >= end:
            continue
        rows.append(appearance)
    return rows


def _appearance_age_question(context: Level3Context, template_id: int, seed: dict[str, str]) -> dict[str, Any]:
    transfer = _specific_transfer(context, seed)
    player = _player_by_id(context, transfer["player_id"])
    if template_id == 70:
        matching = _appearance_window(context, transfer)
        appearance = matching[0] if matching else None
    else:
        previous_transfer = _previous_transfer(context, transfer)
        lower_bound = None
        if previous_transfer and previous_transfer.get("to_club_id") == transfer["from_club_id"]:
            lower_bound = _parse_date(previous_transfer["transfer_date"])
        transfer_date = _parse_date(transfer["transfer_date"])
        matching = []
        for row in context.appearances_by_player_id.get(transfer["player_id"], []):
            appearance_date = _parse_date(row["date"])
            if row["player_club_id"] != transfer["from_club_id"] or appearance_date >= transfer_date:
                continue
            if lower_bound is not None and appearance_date < lower_bound:
                continue
            matching.append(row)
        appearance = matching[-1] if matching else None
    answer = "NULL" if appearance is None else str(_age_on(player, _parse_date(appearance["date"])))
    return _record(context, template_id, {"transfers": transfer, "players": player}, {}, [("transfers", transfer), ("players", player), ("appearance", appearance)], answer, _derived_source("age_on_appearance_date", ["players.date_of_birth", "appearance.date"], "Calculated player age on the matching appearance date."))


def _competition_appearance_question(context: Level3Context, template_id: int, seed: dict[str, str]) -> dict[str, Any]:
    transfer = _specific_transfer(context, seed)
    joined = []
    for appearance in _appearance_window(context, transfer):
        competition = _maybe_competition_by_id(context, appearance.get("competition_id", ""))
        if competition is not None:
            joined.append((appearance, competition))
    fill_rows: dict[str, dict[str, str]] = {"transfers": transfer}
    ground_rows: list[tuple[str, dict[str, str] | None]] = [("transfers", transfer)]
    for appearance, competition in joined:
        ground_rows.extend([("appearance", appearance), ("competitions", competition)])
    if template_id in {72, 73}:
        attribute = "country_name" if template_id == 72 else "confederation"
        fallback_country = None
        for _appearance, competition in joined:
            try:
                fallback_country = _country_for_competition(context, competition)
                break
            except ValueError:
                continue
        if fallback_country is None:
            fallback_country = _club_country(context, _club_by_id(context, transfer["to_club_id"]))
        country = _country_by_seed_or_fallback(context, seed, fallback_country, attribute)
        fill_rows["countries"] = country
        ground_rows.append(("countries", country))
        answer = sum(1 for _appearance, competition in joined if competition.get(attribute) == country[attribute])
        source = _derived_source("count_matches_placeholder", ["appearance.competition_id", f"competitions.{attribute}"], "Counted appearances whose competition matched the placeholder country field.")
    else:
        attribute = "type" if template_id == 74 else "sub_type"
        answer = _format_list(_mode_values([competition.get(attribute, "") for _appearance, competition in joined]))
        source = _derived_source("mode", ["appearance.competition_id", f"competitions.{attribute}"], "Computed the most frequent competition field during the transfer stint.")
    return _record(context, template_id, fill_rows, {}, ground_rows, answer, source)


def _stint_for_transfer(context: Level3Context, transfer: dict[str, str]) -> Stint:
    start = _parse_date(transfer["transfer_date"])
    for stint in context.stints_by_club_id.get(transfer["to_club_id"], []):
        if stint.player_id == transfer["player_id"] and stint.start == start:
            return stint
    next_transfer = _next_transfer(context, transfer)
    return Stint(
        player_id=transfer["player_id"],
        player_name=transfer["player_name"],
        club_id=transfer["to_club_id"],
        club_name=transfer["to_club_name"],
        start=start,
        end=_parse_date(next_transfer["transfer_date"]) if next_transfer else None,
        start_transfer=transfer,
        end_transfer=next_transfer,
    )


def _end_or_open(stint: Stint) -> date:
    return stint.end or OPEN_END


def _overlaps(first: Stint, second: Stint) -> bool:
    return first.start < _end_or_open(second) and second.start < _end_or_open(first)


def _overlap_range(first: Stint, second: Stint) -> tuple[date, date | None] | None:
    if not _overlaps(first, second):
        return None
    start = max(first.start, second.start)
    raw_end = min(_end_or_open(first), _end_or_open(second))
    return start, None if raw_end == OPEN_END else raw_end


def _other_overlapping_stints(context: Level3Context, subject: Stint) -> list[Stint]:
    stints = [stint for stint in context.stints_by_club_id.get(subject.club_id, []) if stint.player_id != subject.player_id and _overlaps(subject, stint)]
    stints.sort(key=lambda stint: (max(stint.start, subject.start), stint.player_name, stint.player_id))
    return stints


def _stints_for_player_name_at_club(context: Level3Context, player_name: str, club_id: str) -> list[Stint]:
    return [stint for stint in context.stints_by_club_id.get(club_id, []) if stint.player_name == player_name]


def _stint_rows(stints: list[Stint]) -> list[tuple[str, dict[str, str] | None]]:
    rows: list[tuple[str, dict[str, str] | None]] = []
    for stint in stints:
        rows.append(("transfers", stint.start_transfer))
        rows.append(("transfers", stint.end_transfer))
    return rows


def _seed_overlap_names(context: Level3Context, seed: dict[str, str], subject: Stint) -> tuple[str, str, str]:
    overlaps = _other_overlapping_stints(context, subject)
    first_other = overlaps[0].player_name if overlaps else subject.player_name
    return (
        _seed_get(seed, "other_player_name") or first_other,
        _seed_get(seed, "player_name_1") or subject.player_name,
        _seed_get(seed, "player_name_2") or first_other,
    )


def _already_at_club(subject: Stint, other: Stint) -> bool:
    return other.start <= subject.start and _end_or_open(other) > subject.start


def _any_named_overlap(context: Level3Context, first_name: str, second_name: str, club_id: str) -> bool:
    for first in _stints_for_player_name_at_club(context, first_name, club_id):
        for second in _stints_for_player_name_at_club(context, second_name, club_id):
            if first.player_id != second.player_id and _overlaps(first, second):
                return True
    return False


def _first_named_overlap_range(context: Level3Context, first_name: str, second_name: str, club_id: str) -> tuple[date, date | None] | None:
    ranges = []
    for first in _stints_for_player_name_at_club(context, first_name, club_id):
        for second in _stints_for_player_name_at_club(context, second_name, club_id):
            if first.player_id == second.player_id:
                continue
            overlap = _overlap_range(first, second)
            if overlap is not None:
                ranges.append(overlap)
    if not ranges:
        return None
    ranges.sort(key=lambda value: value[0])
    return ranges[0]


def _format_overlap_range(overlap: tuple[date, date | None] | None) -> str:
    if overlap is None:
        return "No overlap"
    start, end = overlap
    return f"{start.isoformat()} to {end.isoformat() if end else 'present'}"


def _left_before_arrived(context: Level3Context, first_name: str, second_name: str, club_id: str) -> bool:
    for first in _stints_for_player_name_at_club(context, first_name, club_id):
        if first.end is None:
            continue
        for second in _stints_for_player_name_at_club(context, second_name, club_id):
            if first.player_id != second.player_id and first.end <= second.start:
                return True
    return False


def _overlap_question(context: Level3Context, template_id: int, seed: dict[str, str]) -> dict[str, Any]:
    transfer = _specific_transfer(context, seed)
    subject = _stint_for_transfer(context, transfer)
    overlaps = _other_overlapping_stints(context, subject)
    other_name, player_name_1, player_name_2 = _seed_overlap_names(context, seed, subject)
    extra_values = {"other_player_name": other_name, "player_name_1": player_name_1, "player_name_2": player_name_2}
    ground_rows: list[tuple[str, dict[str, str] | None]] = [("transfers", transfer)]
    ground_rows.extend(("transfers", stint.start_transfer) for stint in overlaps)
    if template_id in {78, 82}:
        ground_rows.extend(_stint_rows(_stints_for_player_name_at_club(context, other_name, subject.club_id)))
    if template_id in {79, 80, 81}:
        ground_rows.extend(_stint_rows(_stints_for_player_name_at_club(context, player_name_1, subject.club_id)))
        ground_rows.extend(_stint_rows(_stints_for_player_name_at_club(context, player_name_2, subject.club_id)))
    if template_id == 76:
        answer = _format_list(list(dict.fromkeys(stint.player_name for stint in overlaps)))
        operation = "list_overlapping_players"
    elif template_id == 77:
        answer = len({stint.player_id for stint in overlaps})
        operation = "count_overlapping_players"
    elif template_id == 78:
        answer = _yes_no(any(stint.player_name == other_name for stint in overlaps))
        operation = "exists_overlap_with_placeholder_player"
    elif template_id == 79:
        answer = _yes_no(_any_named_overlap(context, player_name_1, player_name_2, subject.club_id))
        operation = "named_players_overlap"
    elif template_id == 80:
        answer = _format_overlap_range(_first_named_overlap_range(context, player_name_1, player_name_2, subject.club_id))
        operation = "named_players_overlap_range"
    elif template_id == 81:
        answer = _yes_no(_left_before_arrived(context, player_name_1, player_name_2, subject.club_id))
        operation = "left_before_arrived"
    elif template_id == 82:
        answer = _yes_no(any(stint.player_name == other_name and _already_at_club(subject, stint) for stint in overlaps))
        operation = "already_at_club"
    elif template_id == 83:
        answer = _format_list(list(dict.fromkeys(stint.player_name for stint in overlaps if _already_at_club(subject, stint))))
        operation = "list_already_at_club"
    elif template_id == 84:
        joined_after = [
            stint.player_name
            for stint in context.stints_by_club_id.get(subject.club_id, [])
            if stint.player_id != subject.player_id and subject.start < stint.start < _end_or_open(subject)
        ]
        answer = _format_list(list(dict.fromkeys(joined_after)))
        operation = "list_joined_after_before_next_transfer"
    else:
        raise KeyError(f"Template {template_id}: unsupported overlap question")
    return _record(context, template_id, {"transfers": transfer}, extra_values, ground_rows, answer, _derived_source(operation, ["transfers.player_id", "transfers.transfer_date", "transfers.to_club_id"], "Inferred player club stints from transfer intervals and compared date ranges."))


def _question_dispatch(context: Level3Context, template_id: int, seed: dict[str, str]) -> dict[str, Any]:
    if template_id in {0, 1, 2, 3, 4, 5, 10, 11}:
        return _transfer_aggregate_question(context, template_id, seed, "country_name")
    if template_id in {6, 7, 8, 9}:
        return _transfer_pair_question(context, template_id, seed, "country_name")
    if template_id in {12, 13, 14, 15, 16, 17, 21, 22, 23, 24}:
        attribute = "capital_city" if template_id in {21, 22, 23} else "continent"
        return _transfer_aggregate_question(context, template_id, seed, attribute)
    if template_id in {18, 19, 20}:
        return _transfer_pair_question(context, template_id, seed, "continent")
    if 25 <= template_id <= 33:
        return _game_event_question(context, template_id, seed)
    if 34 <= template_id <= 48:
        return _current_club_country_question(context, template_id, seed)
    if 49 <= template_id <= 53:
        return _current_competition_question(context, template_id, seed)
    if 56 <= template_id <= 63:
        return _current_player_country_question(context, template_id, seed)
    if 64 <= template_id <= 67:
        return _specific_transfer_compare_question(context, template_id, seed)
    if template_id in {68, 69}:
        return _transfer_age_question(context, template_id, seed)
    if template_id in {70, 71}:
        return _appearance_age_question(context, template_id, seed)
    if template_id in {72, 73, 74, 75}:
        return _competition_appearance_question(context, template_id, seed)
    if 76 <= template_id <= 84:
        return _overlap_question(context, template_id, seed)
    raise KeyError(f"No level 3 generator function for template {template_id}")


def _find_default_transfer_seed(context: Level3Context) -> dict[str, str]:
    for transfer in context.table_lists["transfers"]:
        player = _maybe_player_by_id(context, transfer["player_id"])
        from_club = _maybe_club_by_id(context, transfer["from_club_id"])
        to_club = _maybe_club_by_id(context, transfer["to_club_id"])
        if player is None or from_club is None or to_club is None:
            continue
        appearances = context.appearances_by_player_id.get(transfer["player_id"], [])
        has_after = any(appearance["player_club_id"] == transfer["to_club_id"] and appearance["date"] >= transfer["transfer_date"] for appearance in appearances)
        has_before = any(appearance["player_club_id"] == transfer["from_club_id"] and appearance["date"] < transfer["transfer_date"] for appearance in appearances)
        if not (has_after and has_before):
            continue
        player_transfers = context.transfers_by_player_id.get(transfer["player_id"], [])
        try:
            for player_transfer in player_transfers:
                from_transfer_club = _club_by_id(context, player_transfer["from_club_id"])
                to_transfer_club = _club_by_id(context, player_transfer["to_club_id"])
                _club_country(context, from_transfer_club)
                _club_country(context, to_transfer_club)
        except ValueError:
            continue
        to_country = _club_country(context, to_club)
        overlaps = _other_overlapping_stints(context, _stint_for_transfer(context, transfer))
        other_name = overlaps[0].player_name if overlaps else transfer["player_name"]
        return {
            "transfers.player_id": transfer["player_id"],
            "transfers.transfer_date": transfer["transfer_date"],
            "transfers.from_club_id": transfer["from_club_id"],
            "transfers.to_club_id": transfer["to_club_id"],
            "transfer_season": transfer["transfer_season"],
            "target_date": transfer["transfer_date"],
            "countries.country_name": to_country["country_name"],
            "countries.continent": to_country["continent"],
            "countries.capital_city": to_country["capital_city"],
            "countries.confederation": to_country["confederation"],
            "clubs.stadium_seats": to_club["stadium_seats"],
            "stadium_name": to_club["stadium_name"],
            "other_player_name": other_name,
            "player_name_1": transfer["player_name"],
            "player_name_2": other_name,
        }
    raise ValueError("Could not find a default level 3 transfer seed with valid joins")


def _find_default_event_seed(context: Level3Context) -> dict[str, str]:
    for event in context.table_lists["game_events"]:
        club = _maybe_club_by_id(context, event.get("club_id", ""))
        player = _maybe_player_by_id(context, event.get("player_id", ""))
        assist = _maybe_player_by_id(context, event.get("player_assist_id", ""))
        if club is None or player is None:
            continue
        club_country = _club_country(context, club)
        citizenship_country = _maybe_country_by_name(context, player.get("country_of_citizenship", ""))
        birth_country = _maybe_country_by_name(context, player.get("country_of_birth", ""))
        assist_country = _maybe_country_by_name(context, assist.get("country_of_citizenship", "")) if assist is not None else None
        if citizenship_country is None or birth_country is None or assist_country is None:
            continue
        return {
            "game_events.game_event_id": event["game_event_id"],
            "countries.country_name": club_country["country_name"],
            "countries.continent": club_country["continent"],
            "countries.capital_city": birth_country["capital_city"],
            "countries.confederation": club_country["confederation"],
        }
    raise ValueError("Could not find a default level 3 game event seed with valid joins")


def _default_seeds_by_table(context: Level3Context) -> dict[str, list[dict[str, str]]]:
    transfer_seed = _find_default_transfer_seed(context)
    event_seed = _find_default_event_seed(context)
    defaults = {}
    for template in context.templates.values():
        key = _normalize_tables_key(template["tables"])
        if key not in defaults:
            defaults[key] = [event_seed if key.startswith("game_events") else transfer_seed]
    return defaults


def generate_level_3_questions(
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    seed_path: Path = DEFAULT_SEED_PATH,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> list[dict[str, Any]]:
    context = Level3Context.load(template_path=template_path, dataset_dir=dataset_dir)
    seeds_by_table = parse_seed_file(seed_path)
    default_seeds = _default_seeds_by_table(context)
    records = []
    for template_id in sorted(context.templates):
        template = context.templates[template_id]
        tables_key = _normalize_tables_key(template["tables"])
        custom_seeds = seeds_by_table.get(tables_key)
        seeds = custom_seeds or default_seeds.get(tables_key, [])
        start_count = len(records)
        errors: list[str] = []

        for seed in seeds:
            try:
                record = _question_dispatch(context, template_id, seed)
            except ValueError as exc:
                errors.append(f"seed={seed!r}: {exc}")
                continue
            records.append({"question_id": template_id, **record})

        if custom_seeds and len(records) == start_count:
            for seed in default_seeds.get(tables_key, []):
                try:
                    record = _question_dispatch(context, template_id, seed)
                except ValueError as exc:
                    errors.append(f"default seed={seed!r}: {exc}")
                    continue
                records.append({"question_id": template_id, **record})
                print(
                    f"[level_3] template {template_id}: custom seeds failed; used default seed",
                    file=sys.stderr,
                )
                break

        if len(records) == start_count and errors:
            joined_errors = "; ".join(errors)
            raise ValueError(f"Template {template_id} ({tables_key}) could not be generated: {joined_errors}")

    question_ids = [record["question_id"] for record in records]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Level 3 question_id values must be unique. Add explicit IDs for multi-seed templates before generating.")
    return records


def write_questions_json(records: list[dict[str, Any]], output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def generate_and_write_level_3(
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    seed_path: Path = DEFAULT_SEED_PATH,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, Any]]:
    records = generate_level_3_questions(
        template_path=template_path,
        seed_path=seed_path,
        dataset_dir=dataset_dir,
    )
    write_questions_json(records, output_path=output_path)
    return records
