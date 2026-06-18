"""Clean Transfermarkt/player-scores tables for table QA benchmarks.

The script reads raw CSV files from ``raw_data/`` by default and writes the
processed benchmark tables into ``dataset_clean/``. It keeps the requested
output columns, filters players to ``last_season >= 2023``, and filters the
date-based game/event/transfer tables to calendar years 2023 through 2025.
Player-linked tables are aligned to the filtered players table so rows for
players whose last season is 2022 or earlier are removed everywhere.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_INPUT_DIR = Path("raw_data")
DEFAULT_OUTPUT_DIR = Path("dataset_clean")
DATE_FILTER_START = pd.Timestamp("2023-01-01")
DATE_FILTER_END = pd.Timestamp("2025-12-31")
PLAYER_MIN_LAST_SEASON = 2023
NULL_VALUE = "NULL"

CAPITAL_SOURCE_FILENAMES = [
    "all capital cities in the world.csv",
    "all_capital_cities_in_the_world.csv",
    "world_capitals.csv",
    "country_capitals.csv",
]

PLAYER_OUTPUT_COLUMNS = [
    "player_id",
    "name",
    "last_season",
    "player_code",
    "country_of_birth",
    "city_of_birth",
    "country_of_citizenship",
    "date_of_birth",
    "sub_position",
    "position",
    "foot",
    "height_in_cm",
    "agent_name",
]

CLUB_INPUT_COLUMNS = [
    "club_id",
    "club_code",
    "name",
    "domestic_competition_id",
    "stadium_name",
    "stadium_seats",
]

CLUB_OUTPUT_COLUMNS = [
    "club_id",
    "club_code",
    "name",
    "domestic_competition_id",
    "country_name",
    "stadium_name",
    "stadium_seats",
]

COUNTRY_OUTPUT_COLUMNS = [
    "country_id",
    "country_name",
    "country_code",
    "confederation",
    "capital_city",
    "continent",
]

TRANSFER_OUTPUT_COLUMNS = [
    "player_id",
    "transfer_date",
    "transfer_season",
    "from_club_id",
    "to_club_id",
    "from_club_name",
    "to_club_name",
    "player_name",
]

GAMES_OUTPUT_COLUMNS = [
    "game_id",
    "competition_id",
    "season",
    "round",
    "date",
    "home_club_id",
    "away_club_id",
    "home_club_goals",
    "away_club_goals",
]

GAME_EVENTS_OUTPUT_COLUMNS = [
    "game_event_id",
    "date",
    "game_id",
    "minute",
    "type",
    "club_id",
    "club_name",
    "player_id",
    "player_in_id",
    "player_assist_id",
]

COMPETITION_OUTPUT_COLUMNS = [
    "competition_id",
    "competition_code",
    "name",
    "sub_type",
    "type",
    "country_id",
    "country_name",
    "domestic_league_code",
    "confederation",
]

APPEARANCE_OUTPUT_COLUMNS = [
    "appearance_id",
    "game_id",
    "player_id",
    "player_club_id",
    "player_current_club_id",
    "date",
    "player_name",
    "competition_id",
    "yellow_cards",
    "red_cards",
    "goals",
    "assists",
    "minutes_played",
]

def load_csv(input_dir: Path, filename: str) -> pd.DataFrame:
    """Load one raw CSV file and fail with a useful message if it is missing."""
    return load_csv_path(input_dir / filename)


def load_csv_path(path: Path) -> pd.DataFrame:
    """Load one CSV path and fail with a useful message if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    print(f"\nLoading {path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, required_columns: Iterable[str], table_name: str) -> None:
    """Validate that all columns needed for a table are present before cleaning."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        available = ", ".join(df.columns)
        raise ValueError(
            f"{table_name}: missing required columns {missing}. "
            f"Available columns: {available}"
        )


def clean_strings(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Trim text columns and normalize empty strings to pandas missing values."""
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()
            df[column] = df[column].replace("", pd.NA)
    return df


def normalize_id_series(series: pd.Series) -> pd.Series:
    """Normalize ID values so numeric CSV inference does not break joins."""
    numeric = pd.to_numeric(series, errors="coerce")
    normalized = series.astype("string").str.strip()
    numeric_mask = numeric.notna()
    normalized.loc[numeric_mask] = numeric.loc[numeric_mask].astype("Int64").astype("string")
    normalized = normalized.replace("", pd.NA)
    return normalized


def normalize_integer_like_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Write integer-like text values without a trailing .0."""
    cleaned = df.copy()
    for column in columns:
        if column not in cleaned.columns:
            continue

        values = cleaned[column].astype("string").str.strip()
        numeric = pd.to_numeric(values, errors="coerce")
        integer_like = numeric.notna() & numeric.mod(1).eq(0)
        values.loc[integer_like] = numeric.loc[integer_like].astype("Int64").astype("string")
        values = values.replace("<NA>", "")
        cleaned[column] = values.fillna("")
    return cleaned


def format_date_column(series: pd.Series) -> pd.Series:
    """Parse a date-like column and return ISO date strings."""
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").astype("string")


def print_missing_counts(df: pd.DataFrame, table_name: str) -> None:
    """Print missing value counts for the output columns."""
    print(f"{table_name} missing values:")
    missing = df.isna().sum()
    for column, count in missing.items():
        print(f"  {column}: {count}")


def print_date_range(df: pd.DataFrame, table_name: str, column: str) -> None:
    """Print the min/max date for a cleaned date column."""
    dates = pd.to_datetime(df[column], errors="coerce")
    if dates.notna().any():
        print(
            f"{table_name} {column} range: "
            f"{dates.min().date()} to {dates.max().date()}"
        )
    else:
        print(f"{table_name} {column} range: no valid dates")


def save_table(df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    """Save a processed table as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")


def filter_rows_to_players(
    df: pd.DataFrame,
    players: pd.DataFrame,
    table_name: str,
    player_columns: Iterable[str] = ("player_id",),
) -> pd.DataFrame:
    """Keep rows whose non-empty player references appear in cleaned players."""
    require_columns(players, ["player_id"], "filtered players")
    require_columns(df, player_columns, table_name)

    valid_player_ids = set(normalize_id_series(players["player_id"]).dropna())
    keep_mask = pd.Series(True, index=df.index)
    invalid_counts: list[tuple[str, int]] = []

    for column in player_columns:
        player_ids = normalize_id_series(df[column])
        invalid_mask = player_ids.notna() & ~player_ids.isin(valid_player_ids)
        invalid_counts.append((column, int(invalid_mask.sum())))
        keep_mask &= ~invalid_mask

    filtered = df.loc[keep_mask].copy()
    removed = len(df) - len(filtered)
    print(f"{table_name} rows after player alignment: {len(filtered):,} (removed {removed:,})")
    for column, count in invalid_counts:
        print(f"  {column} references outside filtered players: {count:,}")

    return filtered


def clean_players(raw_players: pd.DataFrame) -> pd.DataFrame:
    """Select, clean, and filter player rows."""
    require_columns(raw_players, PLAYER_OUTPUT_COLUMNS, "players")

    print(f"players rows before filtering: {len(raw_players):,}")

    players = raw_players[PLAYER_OUTPUT_COLUMNS].copy()
    players["last_season"] = pd.to_numeric(players["last_season"], errors="coerce").astype("Int64")
    players["height_in_cm"] = pd.to_numeric(players["height_in_cm"], errors="coerce").astype("Int64")
    players["date_of_birth"] = format_date_column(players["date_of_birth"])

    text_columns = [
        "name",
        "player_code",
        "country_of_birth",
        "city_of_birth",
        "country_of_citizenship",
        "sub_position",
        "position",
        "foot",
        "agent_name",
    ]
    players = clean_strings(players, text_columns)

    keep_mask = players["last_season"].ge(PLAYER_MIN_LAST_SEASON).fillna(False)
    players = players.loc[keep_mask].copy()
    print(f"players rows after last_season >= {PLAYER_MIN_LAST_SEASON}: {len(players):,}")

    players = normalize_integer_like_columns(players, ["player_id", "last_season", "height_in_cm"])
    if len(players) > 0:
        seasons = pd.to_numeric(players["last_season"], errors="coerce")
        print(
            "players last_season range after filtering: "
            f"{seasons.min():.0f} to {seasons.max():.0f}"
        )
    print_date_range(players, "players", "date_of_birth")
    print_missing_counts(players, "players")
    return players


def build_players(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Load, clean, and save players.csv."""
    players = clean_players(load_csv(input_dir, "players.csv"))
    save_table(players, output_dir, "players.csv")
    return players


def clean_clubs(
    raw_clubs: pd.DataFrame,
    competitions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select and clean club rows, adding country names from competitions."""
    require_columns(raw_clubs, CLUB_INPUT_COLUMNS, "clubs")

    print(f"clubs rows: {len(raw_clubs):,}")

    clubs = raw_clubs[CLUB_INPUT_COLUMNS].copy()
    clubs["stadium_seats"] = pd.to_numeric(clubs["stadium_seats"], errors="coerce").astype("Int64")
    clubs = clean_strings(
        clubs,
        ["club_code", "name", "domestic_competition_id", "stadium_name"],
    )
    clubs["country_name"] = pd.NA

    if competitions is not None:
        require_columns(competitions, ["competition_id", "country_name"], "cleaned competitions")
        competition_countries = competitions[["competition_id", "country_name"]].copy()
        competition_countries = clean_strings(
            competition_countries,
            ["competition_id", "country_name"],
        )
        competition_countries = competition_countries.drop_duplicates(
            subset=["competition_id"],
            keep="first",
        )
        clubs = clubs.merge(
            competition_countries,
            left_on="domestic_competition_id",
            right_on="competition_id",
            how="left",
            suffixes=("", "_matched"),
        )
        clubs["country_name"] = clubs["country_name_matched"].combine_first(clubs["country_name"])
        clubs = clubs.drop(columns=["competition_id", "country_name_matched"])
        matched_countries = clubs["country_name"].notna().sum()
        print(f"clubs country_name matched from competitions: {matched_countries:,}")
        print(f"clubs without competition country match: {len(clubs) - matched_countries:,}")

    clubs = clubs[CLUB_OUTPUT_COLUMNS]
    clubs = clean_strings(clubs, ["country_name"])
    clubs = normalize_integer_like_columns(clubs, ["club_id", "stadium_seats"])

    print_missing_counts(clubs, "clubs")
    return clubs


def build_clubs(
    input_dir: Path,
    output_dir: Path,
    competitions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load, clean, and save clubs.csv."""
    clubs = clean_clubs(load_csv(input_dir, "clubs.csv"), competitions)
    save_table(clubs, output_dir, "clubs.csv")
    return clubs


def normalize_country_name(country_name: object) -> str:
    """Create a loose key for matching countries across source files."""
    if pd.isna(country_name):
        return ""

    normalized = str(country_name).strip().lower()
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def apply_country_alias(country_key: str) -> str:
    """Map Transfermarkt country spellings to the capital-city file spellings."""
    aliases = {
        "bosnia herzegovina": "bosnia and herzegovina",
        "chinese taipei": "taiwan",
        "czechia": "czech republic",
        "england": "united kingdom",
        "hongkong": "hong kong",
        "ivory coast": "cote d ivoire",
        "korea south": "south korea",
        "northern ireland": "united kingdom",
        "republic of korea": "south korea",
        "scotland": "united kingdom",
        "turkey": "turkiye",
        "wales": "united kingdom",
    }
    return aliases.get(country_key, country_key)


def build_country_name_to_id_lookup(countries: pd.DataFrame) -> pd.DataFrame:
    """Return normalized country names mapped to the cleaned country_id values."""
    require_columns(countries, ["country_id", "country_name"], "cleaned countries")

    lookup = countries[["country_id", "country_name"]].copy()
    lookup = clean_strings(lookup, ["country_name"])
    lookup["country_key"] = (
        lookup["country_name"].map(normalize_country_name).map(apply_country_alias)
    )
    lookup = lookup.dropna(subset=["country_id"])
    lookup = lookup.loc[lookup["country_key"].ne("")].copy()
    lookup = lookup.drop_duplicates(subset=["country_key"], keep="first")
    return lookup[["country_key", "country_id"]]


def pick_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    table_name: str,
    semantic_name: str,
) -> str:
    """Return the first available candidate column name."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    available = ", ".join(df.columns)
    raise ValueError(
        f"{table_name}: missing a {semantic_name} column. "
        f"Tried {list(candidates)}. Available columns: {available}"
    )


def find_capital_source(input_dir: Path, output_dir: Path) -> Path:
    """Find the local CSV that contains world country/capital/continent data."""
    for filename in CAPITAL_SOURCE_FILENAMES:
        path = input_dir / filename
        if path.exists():
            return path

    for path in sorted(input_dir.glob("*capital*.csv")):
        return path

    fallback = output_dir / "countries.csv"
    if fallback.exists():
        return fallback

    expected = ", ".join(CAPITAL_SOURCE_FILENAMES)
    raise FileNotFoundError(
        "Could not find a world capital-city CSV. Add one of these files to "
        f"{input_dir}: {expected}"
    )


def standardize_capitals(raw_capitals: pd.DataFrame) -> pd.DataFrame:
    """Return country_name, capital_city, and continent from common capital CSV schemas."""
    country_column = pick_column(
        raw_capitals,
        ["Country", "country", "country_name", "Country Name", "Name", "name"],
        "capital cities",
        "country name",
    )
    capital_column = pick_column(
        raw_capitals,
        ["Capital City", "capital_city", "Capital", "capital", "Capital Name"],
        "capital cities",
        "capital city",
    )
    continent_column = pick_column(
        raw_capitals,
        ["Continent", "continent", "Continent Name", "Region", "region"],
        "capital cities",
        "continent",
    )

    capitals = raw_capitals[[country_column, capital_column, continent_column]].copy()
    capitals = capitals.rename(
        columns={
            country_column: "country_name",
            capital_column: "capital_city",
            continent_column: "continent",
        }
    )
    capitals = clean_strings(capitals, ["country_name", "capital_city", "continent"])
    capitals = capitals.dropna(subset=["country_name", "capital_city", "continent"])
    capitals["country_key"] = capitals["country_name"].map(normalize_country_name)
    capitals = capitals.loc[capitals["country_key"].ne("")].copy()
    capitals = capitals.drop_duplicates(subset=["country_key"], keep="first")
    capitals = capitals.sort_values(["country_name", "capital_city"], kind="stable")
    return capitals


def clean_countries(raw_countries: pd.DataFrame, raw_capitals: pd.DataFrame) -> pd.DataFrame:
    """Build a country table from all capital-city countries with Transfermarkt metadata."""
    require_columns(
        raw_countries,
        ["country_name", "country_code", "confederation"],
        "countries",
    )

    print(f"countries rows from Transfermarkt: {len(raw_countries):,}")
    print(f"capital-city rows available: {len(raw_capitals):,}")

    capitals = standardize_capitals(raw_capitals)

    transfermarkt = raw_countries[["country_name", "country_code", "confederation"]].copy()
    transfermarkt = clean_strings(
        transfermarkt, ["country_name", "country_code", "confederation"]
    )
    transfermarkt["country_key"] = (
        transfermarkt["country_name"].map(normalize_country_name).map(apply_country_alias)
    )
    transfermarkt = transfermarkt.drop_duplicates(subset=["country_key"], keep="first")
    transfermarkt = transfermarkt.drop(columns=["country_name"])

    countries = capitals.merge(transfermarkt, on="country_key", how="left")
    countries.insert(0, "country_id", range(1, len(countries) + 1))

    metadata_columns = ["country_code", "confederation"]
    unmatched = countries["confederation"].isna().sum()
    countries[metadata_columns] = countries[metadata_columns].fillna(NULL_VALUE)
    countries = countries[COUNTRY_OUTPUT_COLUMNS]
    countries = normalize_integer_like_columns(countries, ["country_id"])

    print(f"countries kept from capital-city file: {len(countries):,}")
    print("country_id generated sequentially from cleaned country order")
    print(f"countries with Transfermarkt confederation: {len(countries) - unmatched:,}")
    print(f"countries with NULL confederation: {unmatched:,}")
    print_missing_counts(countries, "countries")
    return countries


def build_countries(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Load, clean, and save countries.csv."""
    raw_countries = load_csv(input_dir, "countries.csv")
    capital_source = find_capital_source(input_dir, output_dir)
    raw_capitals = load_csv_path(capital_source)
    countries = clean_countries(raw_countries, raw_capitals)
    save_table(countries, output_dir, "countries.csv")
    return countries


def filter_date_window(df: pd.DataFrame, table_name: str, date_column: str) -> pd.DataFrame:
    """Keep rows whose date column falls in the requested calendar window."""
    filtered = df.copy()
    filtered[date_column] = pd.to_datetime(filtered[date_column], errors="coerce")
    print_date_range(filtered, f"{table_name} raw", date_column)
    keep_mask = filtered[date_column].between(DATE_FILTER_START, DATE_FILTER_END, inclusive="both")
    filtered = filtered.loc[keep_mask].copy()
    filtered[date_column] = filtered[date_column].dt.strftime("%Y-%m-%d").astype("string")
    print_date_range(filtered, table_name, date_column)
    return filtered


def clean_transfers(raw_transfers: pd.DataFrame, players: pd.DataFrame | None = None) -> pd.DataFrame:
    """Select, clean, date-filter, and optionally align transfer rows to players."""
    require_columns(raw_transfers, TRANSFER_OUTPUT_COLUMNS, "transfers")

    print(f"transfers rows before filtering: {len(raw_transfers):,}")

    transfers = raw_transfers[TRANSFER_OUTPUT_COLUMNS].copy()
    transfers = filter_date_window(transfers, "transfers", "transfer_date")
    transfers = clean_strings(
        transfers,
        ["transfer_season", "from_club_name", "to_club_name", "player_name"],
    )
    if players is not None:
        transfers = filter_rows_to_players(transfers, players, "transfers")

    transfers = normalize_integer_like_columns(
        transfers,
        ["player_id", "from_club_id", "to_club_id"],
    )

    print(f"transfers rows after filtering: {len(transfers):,}")
    print_missing_counts(transfers, "transfers")
    return transfers


def build_transfers(
    input_dir: Path,
    output_dir: Path,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load, clean, and save transfers.csv."""
    transfers = clean_transfers(load_csv(input_dir, "transfers.csv"), players)
    save_table(transfers, output_dir, "transfers.csv")
    return transfers


def clean_games(raw_games: pd.DataFrame) -> pd.DataFrame:
    """Select, clean, and filter game rows for seasons/dates 2023 through 2025."""
    require_columns(raw_games, GAMES_OUTPUT_COLUMNS, "games")

    print(f"games rows before filtering: {len(raw_games):,}")

    games = raw_games[GAMES_OUTPUT_COLUMNS].copy()
    games["season"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    print_date_range(games, "games raw", "date")

    season_mask = games["season"].between(2023, 2025, inclusive="both").fillna(False)
    date_mask = games["date"].between(DATE_FILTER_START, DATE_FILTER_END, inclusive="both")
    games = games.loc[season_mask & date_mask].copy()
    games["date"] = games["date"].dt.strftime("%Y-%m-%d").astype("string")

    games = clean_strings(
        games,
        [
            "competition_id",
            "round",
        ],
    )
    games = normalize_integer_like_columns(
        games,
        [
            "game_id",
            "season",
            "home_club_id",
            "away_club_id",
            "home_club_goals",
            "away_club_goals",
        ],
    )

    print(f"games rows after filtering: {len(games):,}")
    print_date_range(games, "games", "date")
    print_missing_counts(games, "games")
    return games


def build_games(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Load, clean, and save games.csv."""
    games = clean_games(load_csv(input_dir, "games.csv"))
    save_table(games, output_dir, "games.csv")
    return games


def clean_game_events(
    raw_events: pd.DataFrame,
    games: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select, clean, date-filter, and optionally align game-event rows."""
    require_columns(raw_events, GAME_EVENTS_OUTPUT_COLUMNS, "game_events")

    print(f"game_events rows before filtering: {len(raw_events):,}")

    events = raw_events[GAME_EVENTS_OUTPUT_COLUMNS].copy()
    events = filter_date_window(events, "game_events", "date")

    if games is not None:
        valid_game_ids = set(normalize_id_series(games["game_id"]).dropna())
        event_game_ids = normalize_id_series(events["game_id"])
        before = len(events)
        events = events.loc[event_game_ids.isin(valid_game_ids)].copy()
        print(f"game_events rows after game_id alignment: {len(events):,} (removed {before - len(events):,})")

    if players is not None:
        events = filter_rows_to_players(
            events,
            players,
            "game_events",
            ["player_id", "player_in_id", "player_assist_id"],
        )

    events = clean_strings(events, ["game_event_id", "type", "club_name"])
    events = normalize_integer_like_columns(
        events,
        [
            "game_id",
            "minute",
            "club_id",
            "player_id",
            "player_in_id",
            "player_assist_id",
        ],
    )

    print(f"game_events rows after filtering: {len(events):,}")
    print_missing_counts(events, "game_events")
    return events


def build_game_events(
    input_dir: Path,
    output_dir: Path,
    games: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load, clean, and save game_events.csv."""
    events = clean_game_events(load_csv(input_dir, "game_events.csv"), games, players)
    save_table(events, output_dir, "game_events.csv")
    return events


def clean_competitions(
    raw_competitions: pd.DataFrame,
    countries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select and clean competition rows, updating country_id from country_name."""
    require_columns(raw_competitions, COMPETITION_OUTPUT_COLUMNS, "competitions")

    print(f"competitions rows: {len(raw_competitions):,}")

    competitions = raw_competitions[COMPETITION_OUTPUT_COLUMNS].copy()
    competitions = clean_strings(
        competitions,
        [
            "competition_id",
            "competition_code",
            "name",
            "sub_type",
            "type",
            "country_name",
            "domestic_league_code",
            "confederation",
        ],
    )
    competitions["competition_code"] = competitions["competition_id"]
    if countries is not None:
        country_lookup = build_country_name_to_id_lookup(countries).rename(
            columns={"country_id": "clean_country_id"}
        )
        competitions["country_key"] = (
            competitions["country_name"].map(normalize_country_name).map(apply_country_alias)
        )
        competitions = competitions.merge(country_lookup, on="country_key", how="left")
        matched_country_ids = competitions["clean_country_id"].notna().sum()
        competitions["country_id"] = competitions["clean_country_id"].fillna(
            competitions["country_id"]
        )
        competitions = competitions.drop(columns=["country_key", "clean_country_id"])
        print(f"competitions country_id updated from country_name: {matched_country_ids:,}")
        print(
            "competitions without country_name match: "
            f"{len(competitions) - matched_country_ids:,}"
        )

    competitions = normalize_integer_like_columns(competitions, ["country_id"])
    print_missing_counts(competitions, "competitions")
    return competitions


def build_competitions(
    input_dir: Path,
    output_dir: Path,
    countries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load, clean, and save competitions.csv."""
    competitions = clean_competitions(load_csv(input_dir, "competitions.csv"), countries)
    save_table(competitions, output_dir, "competitions.csv")
    return competitions


def clean_appearances(
    raw_appearances: pd.DataFrame,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select, clean, and optionally align appearance rows to players."""
    require_columns(raw_appearances, APPEARANCE_OUTPUT_COLUMNS, "appearances")

    print(f"appearances rows: {len(raw_appearances):,}")

    appearances = raw_appearances[APPEARANCE_OUTPUT_COLUMNS].copy()
    appearances["date"] = format_date_column(appearances["date"])
    appearances = clean_strings(appearances, ["appearance_id", "player_name", "competition_id"])
    if players is not None:
        appearances = filter_rows_to_players(appearances, players, "appearances")

    appearances = normalize_integer_like_columns(
        appearances,
        [
            "game_id",
            "player_id",
            "player_club_id",
            "player_current_club_id",
            "yellow_cards",
            "red_cards",
            "goals",
            "assists",
            "minutes_played",
        ],
    )
    print_date_range(appearances, "appearances", "date")
    print_missing_counts(appearances, "appearances")
    return appearances


def build_appearances(
    input_dir: Path,
    output_dir: Path,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load, clean, and save appearances.csv."""
    appearances = clean_appearances(load_csv(input_dir, "appearances.csv"), players)
    save_table(appearances, output_dir, "appearances.csv")
    return appearances


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments while keeping requested defaults."""
    parser = argparse.ArgumentParser(
        description="Clean Transfermarkt/player-scores CSV files for benchmark tables."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing raw CSV files. Defaults to raw_data/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where processed CSV files will be written. Defaults to dataset_clean/.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full cleaning pipeline."""
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    players = build_players(input_dir, output_dir)
    countries = build_countries(input_dir, output_dir)
    competitions = build_competitions(input_dir, output_dir, countries)
    build_clubs(input_dir, output_dir, competitions)
    build_transfers(input_dir, output_dir, players)
    games = build_games(input_dir, output_dir)
    build_game_events(input_dir, output_dir, games, players)
    build_appearances(input_dir, output_dir, players)

    print("\nDone.")


if __name__ == "__main__":
    main()
