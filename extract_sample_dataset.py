"""Extract a player-centered sample dataset from benchmark CSV files.

By default this script extracts an Abdoulay Diaby sample from ``dataset_clean/``
into ``dataset_sample/``. Passing ``--input-dir dataset_counter`` without an
explicit output directory writes ``dataset_counter_sample/``. The output
directory is overwritten before CSVs are written. The sample keeps one selected
row in players.csv, direct player-linked rows, related dimension rows, and
three transfer-only comparison players: same club with overlapping time, same
club without overlapping time, and no shared transfer club.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CLEAN_INPUT_DIR = Path("dataset_clean")
DEFAULT_CLEAN_OUTPUT_DIR = Path("dataset_sample")
DEFAULT_COUNTER_INPUT_DIR = Path("dataset_counter")
DEFAULT_COUNTER_OUTPUT_DIR = Path("dataset_counter_sample")
DEFAULT_INPUT_DIR = DEFAULT_CLEAN_INPUT_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_CLEAN_OUTPUT_DIR
DEFAULT_PLAYER_NAME = "Abdoulay Diaby"
DEFAULT_TRANSFER_EXAMPLE_PLAYERS = 3
TRANSFER_WINDOW_END = pd.Timestamp("2025-12-31")
WITHOUT_CLUB_ID = "515"
WITHOUT_CLUB_NAME = "Without Club"

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

PLAYER_EVENT_COLUMNS = ["player_id", "player_in_id", "player_assist_id"]


def resolve_output_dir(input_dir: Path, output_dir: Path | None) -> Path:
    """Resolve the default sample directory for clean or counter datasets."""
    if output_dir is not None:
        return output_dir
    if input_dir.name == DEFAULT_COUNTER_INPUT_DIR.name:
        return DEFAULT_COUNTER_OUTPUT_DIR
    return DEFAULT_CLEAN_OUTPUT_DIR


def load_table(input_dir: Path, table_name: str) -> pd.DataFrame:
    """Load one cleaned CSV as strings so IDs stay join-safe."""
    path = input_dir / TABLE_FILENAMES[table_name]
    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")
    return pd.read_csv(path, dtype="string").fillna("")


def normalize_id_series(series: pd.Series) -> pd.Series:
    """Normalize ID values across numeric-looking CSV strings."""
    values = series.astype("string").str.strip()
    numeric = pd.to_numeric(values, errors="coerce")
    integer_like = numeric.notna() & numeric.mod(1).eq(0)
    values.loc[integer_like] = numeric.loc[integer_like].astype("Int64").astype("string")
    return values.fillna("")


def value_set(df: pd.DataFrame, columns: Iterable[str]) -> set[str]:
    """Collect normalized non-empty values from one or more columns."""
    values: set[str] = set()
    for column in columns:
        if column in df.columns:
            values.update(normalize_id_series(df[column]).loc[lambda s: s.ne("")])
    return values


def text_set(df: pd.DataFrame, columns: Iterable[str]) -> set[str]:
    """Collect stripped non-empty text values from one or more columns."""
    values: set[str] = set()
    for column in columns:
        if column in df.columns:
            values.update(df[column].astype("string").str.strip().loc[lambda s: s.ne("")])
    return values


def ids_in(series: pd.Series, ids: set[str]) -> pd.Series:
    """Return a boolean mask for normalized IDs contained in ids."""
    return normalize_id_series(series).isin(ids)


def find_player(players: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Return rows for an exact case-insensitive player name match."""
    names = players["name"].astype("string").str.strip().str.casefold()
    selected = players.loc[names.eq(player_name.strip().casefold())].copy()
    if selected.empty:
        raise ValueError(f"No player named {player_name!r} found in players.csv")
    return selected


def sorted_unique_player_ids(transfers: pd.DataFrame) -> list[str]:
    """Return deterministic player IDs from transfer rows."""
    ordered = transfers.assign(_player_id=normalize_id_series(transfers["player_id"]))
    ordered = ordered.sort_values(["player_name", "_player_id", "transfer_date"], kind="stable")
    return ordered["_player_id"].drop_duplicates().loc[lambda s: s.ne("")].tolist()


def is_real_club(club_id: str, club_name: str = "") -> bool:
    """Return True for actual clubs, excluding the synthetic Without Club row."""
    return club_id not in {"", WITHOUT_CLUB_ID} and club_name != WITHOUT_CLUB_NAME


def transfer_club_set(transfers: pd.DataFrame) -> set[str]:
    """Collect real club IDs from from/to transfer club columns."""
    clubs: set[str] = set()
    for id_column, name_column in [
        ("from_club_id", "from_club_name"),
        ("to_club_id", "to_club_name"),
    ]:
        club_ids = normalize_id_series(transfers[id_column])
        club_names = transfers[name_column].astype("string").str.strip()
        for club_id, club_name in zip(club_ids, club_names, strict=False):
            if is_real_club(club_id, club_name):
                clubs.add(club_id)
    return clubs


def build_transfer_spells(transfers: pd.DataFrame) -> pd.DataFrame:
    """Infer to-club spells from transfer dates."""
    columns = ["player_id", "player_name", "club_id", "club_name", "start", "end"]
    if transfers.empty:
        return pd.DataFrame(columns=columns)

    ordered = transfers.copy()
    ordered["_player_id"] = normalize_id_series(ordered["player_id"])
    ordered["_to_club_id"] = normalize_id_series(ordered["to_club_id"])
    ordered["_transfer_date"] = pd.to_datetime(ordered["transfer_date"], errors="coerce")
    ordered = ordered.sort_values(["_player_id", "_transfer_date", "player_name"], kind="stable")

    spells: list[dict[str, object]] = []
    for player_id, group in ordered.groupby("_player_id", sort=False):
        group = group.reset_index(drop=True)
        for index, row in group.iterrows():
            start = row["_transfer_date"]
            if pd.isna(start):
                continue

            club_id = row["_to_club_id"]
            club_name = str(row["to_club_name"]).strip()
            if not is_real_club(club_id, club_name):
                continue

            if index + 1 < len(group):
                end = group.loc[index + 1, "_transfer_date"]
            else:
                end = TRANSFER_WINDOW_END

            if pd.isna(end) or end <= start:
                continue

            spells.append(
                {
                    "player_id": player_id,
                    "player_name": row["player_name"],
                    "club_id": club_id,
                    "club_name": club_name,
                    "start": start,
                    "end": end,
                }
            )

    return pd.DataFrame(spells, columns=columns)


def intervals_overlap(left: pd.Series, right: pd.Series) -> bool:
    """Return True when two half-open date intervals overlap."""
    return left["start"] < right["end"] and right["start"] < left["end"]


def player_has_overlapping_shared_spell(
    candidate_spells: pd.DataFrame,
    selected_spells: pd.DataFrame,
) -> bool:
    """Check whether candidate has any same-club spell overlapping selected player."""
    for _, candidate_spell in candidate_spells.iterrows():
        matching_selected = selected_spells.loc[
            selected_spells["club_id"].eq(candidate_spell["club_id"])
        ]
        for _, selected_spell in matching_selected.iterrows():
            if intervals_overlap(candidate_spell, selected_spell):
                return True
    return False


def choose_transfer_example_player_ids(
    transfers: pd.DataFrame,
    transfers_all: pd.DataFrame,
    player_ids: set[str],
    example_player_count: int,
) -> list[str]:
    """Choose overlap, same-club/no-overlap, and no-shared-club examples."""
    if example_player_count <= 0:
        return []

    selected_spells = build_transfer_spells(transfers)
    if selected_spells.empty:
        return []

    all_spells = build_transfer_spells(transfers_all)
    selected_spell_clubs = set(selected_spells["club_id"])
    selected_transfer_clubs = transfer_club_set(transfers)
    candidate_spells = all_spells.loc[
        ~all_spells["player_id"].isin(player_ids)
        & all_spells["club_id"].isin(selected_spell_clubs)
    ].copy()
    candidate_spells = candidate_spells.sort_values(
        ["player_name", "player_id", "club_name", "start"],
        kind="stable",
    )

    chosen: list[str] = []

    for player_id in candidate_spells["player_id"].drop_duplicates():
        spells = candidate_spells.loc[candidate_spells["player_id"].eq(player_id)]
        if player_has_overlapping_shared_spell(spells, selected_spells):
            chosen.append(player_id)
            break

    if len(chosen) < example_player_count:
        for player_id in candidate_spells["player_id"].drop_duplicates():
            if player_id in chosen:
                continue

            spells = candidate_spells.loc[candidate_spells["player_id"].eq(player_id)]
            if not player_has_overlapping_shared_spell(spells, selected_spells):
                chosen.append(player_id)
                break

    if len(chosen) < example_player_count:
        candidates = transfers_all.loc[
            ~ids_in(transfers_all["player_id"], player_ids | set(chosen))
        ].copy()
        for player_id in sorted_unique_player_ids(candidates):
            player_transfers = candidates.loc[ids_in(candidates["player_id"], {player_id})]
            if transfer_club_set(player_transfers).isdisjoint(selected_transfer_clubs):
                chosen.append(player_id)
                break

    return chosen[:example_player_count]


def add_transfer_examples(
    transfers: pd.DataFrame,
    transfers_all: pd.DataFrame,
    player_ids: set[str],
    example_player_count: int,
) -> pd.DataFrame:
    """Add transfer-only players for overlap, no-overlap, and no-shared examples."""
    if example_player_count <= 0 or transfers.empty:
        return transfers

    chosen_player_ids = choose_transfer_example_player_ids(
        transfers,
        transfers_all,
        player_ids,
        example_player_count,
    )
    if not chosen_player_ids:
        return transfers

    examples = transfers_all.loc[ids_in(transfers_all["player_id"], set(chosen_player_ids))].copy()
    return pd.concat([transfers, examples], ignore_index=True).drop_duplicates()


def extract_sample(
    input_dir: Path,
    output_dir: Path,
    player_name: str,
    transfer_example_players: int = DEFAULT_TRANSFER_EXAMPLE_PLAYERS,
) -> dict[str, pd.DataFrame]:
    """Build all sample tables for one player."""
    players_all = load_table(input_dir, "players")
    appearances_all = load_table(input_dir, "appearances")
    transfers_all = load_table(input_dir, "transfers")
    game_events_all = load_table(input_dir, "game_events")
    games_all = load_table(input_dir, "games")
    clubs_all = load_table(input_dir, "clubs")
    competitions_all = load_table(input_dir, "competitions")
    countries_all = load_table(input_dir, "countries")

    players = find_player(players_all, player_name)
    player_ids = value_set(players, ["player_id"])

    appearances = appearances_all.loc[ids_in(appearances_all["player_id"], player_ids)].copy()
    transfers = transfers_all.loc[ids_in(transfers_all["player_id"], player_ids)].copy()
    transfers = add_transfer_examples(
        transfers,
        transfers_all,
        player_ids,
        transfer_example_players,
    )

    event_mask = pd.Series(False, index=game_events_all.index)
    for column in PLAYER_EVENT_COLUMNS:
        event_mask |= ids_in(game_events_all[column], player_ids)
    game_events = game_events_all.loc[event_mask].copy()

    game_ids = value_set(appearances, ["game_id"]) | value_set(game_events, ["game_id"])
    games = games_all.loc[ids_in(games_all["game_id"], game_ids)].copy()

    club_ids = (
        value_set(appearances, ["player_club_id", "player_current_club_id"])
        | value_set(transfers, ["from_club_id", "to_club_id"])
        | value_set(game_events, ["club_id"])
        | value_set(games, ["home_club_id", "away_club_id"])
    )
    clubs = clubs_all.loc[ids_in(clubs_all["club_id"], club_ids)].copy()

    competition_ids = (
        value_set(appearances, ["competition_id"])
        | value_set(games, ["competition_id"])
        | value_set(clubs, ["domestic_competition_id"])
    )
    competitions = competitions_all.loc[
        normalize_id_series(competitions_all["competition_id"]).isin(competition_ids)
    ].copy()

    country_names = (
        text_set(players, ["country_of_birth", "country_of_citizenship"])
        | text_set(clubs, ["country_name"])
        | text_set(competitions, ["country_name"])
    )
    countries = countries_all.loc[countries_all["country_name"].isin(country_names)].copy()

    return {
        "players": players,
        "countries": countries,
        "clubs": clubs,
        "competitions": competitions,
        "games": games,
        "game_events": game_events,
        "appearances": appearances,
        "transfers": transfers,
    }


def prepare_output_dir(input_dir: Path, output_dir: Path) -> None:
    """Overwrite the output directory after basic safety checks."""
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()
    cwd_resolved = Path.cwd().resolve()

    if output_resolved == input_resolved:
        raise ValueError(f"Refusing to overwrite input directory: {output_dir}")
    if output_resolved == cwd_resolved:
        raise ValueError(f"Refusing to overwrite current working directory: {output_dir}")
    if output_resolved.parent == output_resolved:
        raise ValueError(f"Refusing to overwrite filesystem root: {output_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def save_sample(
    sample_tables: dict[str, pd.DataFrame],
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Write sample CSV files and print row counts."""
    prepare_output_dir(input_dir, output_dir)
    for table_name in TABLE_FILENAMES:
        df = sample_tables[table_name]
        path = output_dir / TABLE_FILENAMES[table_name]
        df.to_csv(path, index=False)
        print(f"{table_name}: {len(df):,} rows -> {path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract a player-centered sample dataset from dataset_clean or dataset_counter."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing cleaned CSV files. Defaults to dataset_clean/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where sample CSV files will be written. Defaults to "
            "dataset_counter_sample/ for dataset_counter input, otherwise dataset_sample/."
        ),
    )
    parser.add_argument(
        "--player-name",
        default=DEFAULT_PLAYER_NAME,
        help="Exact player name to extract. Defaults to Abdoulay Diaby.",
    )
    parser.add_argument(
        "--transfer-example-players",
        type=int,
        default=DEFAULT_TRANSFER_EXAMPLE_PLAYERS,
        help=(
            "Number of extra transfer-only comparison players to include. "
            "Defaults to 3: same-club overlap, same-club no-overlap, and no-shared-club."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run sample extraction."""
    args = parse_args()
    output_dir = resolve_output_dir(args.input_dir, args.output_dir)

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")

    sample_tables = extract_sample(
        args.input_dir,
        output_dir,
        args.player_name,
        args.transfer_example_players,
    )
    save_sample(sample_tables, args.input_dir, output_dir)


if __name__ == "__main__":
    main()
