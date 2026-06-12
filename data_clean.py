"""Clean selected Transfermarkt/player-scores tables for table QA benchmarks.

The script reads raw CSV files from ``raw_data/`` by default and writes the
processed benchmark tables into ``dataset_clean/``. The full pipeline also
applies the post-cleaning filters that previously lived in
``data_clean_update.py``:

1. Keep only players with both ``country_of_birth`` and ``city_of_birth``.
2. Keep only transfer rows whose ``player_id`` remains in the filtered players
   table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_INPUT_DIR = Path("raw_data")
DEFAULT_OUTPUT_DIR = Path("dataset_clean")

PLAYER_OUTPUT_COLUMNS = [
    "player_id",
    "name",
    "last_season",
    "country_of_birth",
    "city_of_birth",
    "country_of_citizenship",
    "date_of_birth",
    "position",
    "height",
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
    *CLUB_INPUT_COLUMNS,
    "country_name",
]

COUNTRY_OUTPUT_COLUMNS = [
    "country_id",
    "country_name",
    "country_code",
    "confederation",
    "continent",
    "capital_city",
]

NULL_VALUE = "NULL"

# Transfermarkt club files occasionally use a competition ID that differs from
# the country_code in countries.csv.
COMPETITION_COUNTRY_CODE_ALIASES = {
    "COL1": "COLP",
}

TRANSFER_OUTPUT_COLUMNS = [
    "transfer_id",
    "player_id",
    "transfer_date",
    "from_club_id",
    "to_club_id",
    "from_club_name",
    "to_club_name",
    "player_name",
]


def load_csv(input_dir: Path, filename: str) -> pd.DataFrame:
    """Load one raw CSV file and fail with a useful message if it is missing."""
    path = input_dir / filename
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


def valid_text_mask(series: pd.Series) -> pd.Series:
    """Return True for non-null, non-blank values."""
    text = series.astype("string").str.strip()
    return (series.notna() & text.notna() & text.ne("")).fillna(False)


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


def filter_players_with_birth_location(players: pd.DataFrame) -> pd.DataFrame:
    """Keep only players that have both country and city of birth."""
    require_columns(players, ["player_id", "country_of_birth", "city_of_birth"], "players")

    has_country = valid_text_mask(players["country_of_birth"])
    has_city = valid_text_mask(players["city_of_birth"])
    filtered = players.loc[has_country & has_city].copy()

    print("\nPlayer birth-location filtering")
    print(f"  rows before: {len(players):,}")
    print(f"  missing country_of_birth: {(~has_country).sum():,}")
    print(f"  missing city_of_birth: {(~has_city).sum():,}")
    print(f"  rows after: {len(filtered):,}")
    print(f"  rows removed: {len(players) - len(filtered):,}")

    return normalize_integer_like_columns(filtered, ["player_id", "last_season", "height"])


def clean_players(raw_players: pd.DataFrame) -> pd.DataFrame:
    """Select, clean, and filter player rows."""
    required_columns = [
        "player_id",
        "name",
        "last_season",
        "country_of_birth",
        "city_of_birth",
        "country_of_citizenship",
        "date_of_birth",
        "position",
        "height_in_cm",
    ]
    require_columns(raw_players, required_columns, "players")

    print(f"players rows before filtering: {len(raw_players):,}")

    players = raw_players[required_columns].copy()
    players = players.rename(columns={"height_in_cm": "height"})

    # Convert numeric and date fields before filtering and saving.
    players["last_season"] = pd.to_numeric(players["last_season"], errors="coerce").astype("Int64")
    players["height"] = pd.to_numeric(players["height"], errors="coerce").astype("Int64")
    players["date_of_birth"] = format_date_column(players["date_of_birth"])

    text_columns = [
        "name",
        "country_of_birth",
        "city_of_birth",
        "country_of_citizenship",
        "position",
    ]
    players = clean_strings(players, text_columns)

    # Keep only players active in the requested last-season window.
    players = players[
        players["last_season"].between(2023, 2026, inclusive="both")
    ].copy()
    print(f"players rows after season filtering: {len(players):,}")

    players = players[PLAYER_OUTPUT_COLUMNS]
    players = filter_players_with_birth_location(players)
    if players["last_season"].notna().any():
        print(
            "players last_season range after filtering: "
            f"{players['last_season'].min()} to {players['last_season'].max()}"
        )
    print_date_range(players, "players", "date_of_birth")
    print_missing_counts(players, "players")
    return players


def build_players(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Load, clean, and save players.csv."""
    raw_players = load_csv(input_dir, "players.csv")
    players = clean_players(raw_players)
    save_table(players, output_dir, "players.csv")
    return players


def build_country_code_lookup(raw_countries: pd.DataFrame) -> pd.DataFrame:
    """Return country_code to country_name rows for club-country matching."""
    require_columns(raw_countries, ["country_code", "country_name"], "countries")

    lookup = raw_countries[["country_code", "country_name"]].copy()
    lookup = clean_strings(lookup, ["country_code", "country_name"])
    lookup = lookup.dropna(subset=["country_code"])
    lookup = lookup.drop_duplicates(subset=["country_code"], keep="first")
    return lookup


def clean_clubs(raw_clubs: pd.DataFrame, raw_countries: pd.DataFrame) -> pd.DataFrame:
    """Select and clean club rows, including the matched country name."""
    require_columns(raw_clubs, CLUB_INPUT_COLUMNS, "clubs")

    print(f"clubs rows: {len(raw_clubs):,}")

    clubs = raw_clubs[CLUB_INPUT_COLUMNS].copy()
    clubs["stadium_seats"] = pd.to_numeric(clubs["stadium_seats"], errors="coerce").astype("Int64")

    text_columns = [
        "club_code",
        "name",
        "domestic_competition_id",
        "stadium_name",
    ]
    clubs = clean_strings(clubs, text_columns)

    country_lookup = build_country_code_lookup(raw_countries).rename(
        columns={"country_code": "country_code_key"}
    )
    clubs["country_code_key"] = clubs["domestic_competition_id"].replace(
        COMPETITION_COUNTRY_CODE_ALIASES
    )
    clubs = clubs.merge(country_lookup, on="country_code_key", how="left")
    matched_countries = clubs["country_name"].notna().sum()
    clubs["country_name"] = clubs["country_name"].fillna(NULL_VALUE)
    clubs = clubs.drop(columns=["country_code_key"])
    clubs = clubs[CLUB_OUTPUT_COLUMNS]

    print(f"clubs matched to country_name: {matched_countries:,}")
    print(f"clubs without matched country_name: {len(clubs) - matched_countries:,}")
    print_missing_counts(clubs, "clubs")
    return clubs


def build_clubs(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Load, clean, and save clubs.csv."""
    raw_clubs = load_csv(input_dir, "clubs.csv")
    raw_countries = load_csv(input_dir, "countries.csv")
    clubs = clean_clubs(raw_clubs, raw_countries)
    save_table(clubs, output_dir, "clubs.csv")
    return clubs


def normalize_country_name(country_name: object) -> str:
    """Create a loose key for matching countries across source files."""
    if pd.isna(country_name):
        return ""

    normalized = str(country_name).strip().lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def apply_country_alias(country_key: str) -> str:
    """Map Transfermarkt country spellings to the capital-city file spellings."""
    aliases = {
        "north macedonia": "north macedonia",
        "myanmar": "myanmar",
        "chinese taipei": "taiwan",
        "turkiye": "turkiye",
        "england": "united kingdom",
        "hongkong": "hong kong",
        "bosnia herzegovina": "bosnia and herzegovina",
        "korea south": "south korea",
    }
    return aliases.get(country_key, country_key)


def clean_countries(raw_countries: pd.DataFrame, raw_capitals: pd.DataFrame) -> pd.DataFrame:
    """Build a country table from all capital-city countries with Transfermarkt metadata."""
    require_columns(
        raw_countries,
        ["country_name", "country_code", "confederation"],
        "countries",
    )
    require_columns(raw_capitals, ["Capital City", "Country", "Continent"], "capital cities")

    print(f"countries rows from Transfermarkt: {len(raw_countries):,}")
    print(f"capital-city rows available: {len(raw_capitals):,}")

    # The capital-city file is now the source of truth for country coverage.
    capitals = raw_capitals[["Country", "Capital City", "Continent"]].copy()
    capitals = capitals.rename(
        columns={
            "Country": "country_name",
            "Capital City": "capital_city",
            "Continent": "continent",
        }
    )
    capitals = clean_strings(capitals, ["country_name", "capital_city", "continent"])
    capitals["country_key"] = capitals["country_name"].map(normalize_country_name)
    capitals = capitals.drop_duplicates(subset=["country_key"], keep="first")

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

    print(f"countries kept from capital-city file: {len(countries):,}")
    print("country_id generated sequentially from cleaned country order")
    print(f"countries with Transfermarkt confederation: {len(countries) - unmatched:,}")
    print(f"countries with NULL confederation: {unmatched:,}")
    print_missing_counts(countries, "countries")
    return countries


def build_countries(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Load, clean, and save countries.csv."""
    raw_countries = load_csv(input_dir, "countries.csv")
    raw_capitals = load_csv(input_dir, "all capital cities in the world.csv")
    countries = clean_countries(raw_countries, raw_capitals)
    save_table(countries, output_dir, "countries.csv")
    return countries


def clean_transfers(raw_transfers: pd.DataFrame) -> pd.DataFrame:
    """Select, clean, date-filter, and identify transfer rows."""
    base_columns = [
        "player_id",
        "transfer_date",
        "from_club_id",
        "to_club_id",
        "from_club_name",
        "to_club_name",
        "player_name",
    ]
    require_columns(raw_transfers, base_columns, "transfers")

    print(f"transfers rows before filtering: {len(raw_transfers):,}")

    transfers = raw_transfers.copy()
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    print_date_range(transfers, "transfers raw", "transfer_date")

    # Calendar-year filtering is intentional. Do not filter by transfer_season.
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2025-12-31")
    transfers = transfers[
        transfers["transfer_date"].between(start_date, end_date, inclusive="both")
    ].copy()

    if "transfer_id" not in transfers.columns:
        transfers.insert(0, "transfer_id", range(1, len(transfers) + 1))
        print("transfers: created synthetic transfer_id")

    transfers["transfer_date"] = transfers["transfer_date"].dt.strftime("%Y-%m-%d").astype("string")

    text_columns = [
        "from_club_name",
        "to_club_name",
        "player_name",
    ]
    transfers = clean_strings(transfers, text_columns)

    transfers = transfers[TRANSFER_OUTPUT_COLUMNS]

    print(f"transfers rows after filtering: {len(transfers):,}")
    print_date_range(transfers, "transfers", "transfer_date")
    print_missing_counts(transfers, "transfers")
    return transfers


def filter_transfers_to_players(
    transfers: pd.DataFrame, players: pd.DataFrame
) -> pd.DataFrame:
    """Keep only transfers whose player_id appears in the filtered players table."""
    require_columns(transfers, ["player_id"], "transfers.csv")
    require_columns(players, ["player_id"], "filtered players.csv")

    valid_player_ids = set(normalize_id_series(players["player_id"]).dropna())
    transfer_player_ids = normalize_id_series(transfers["player_id"])
    keep_mask = transfer_player_ids.isin(valid_player_ids)
    filtered = transfers.loc[keep_mask].copy()

    unmatched_ids = sorted(transfer_player_ids.loc[~keep_mask].dropna().unique())[:20]

    print("\nTransfer player filtering")
    print(f"  rows before: {len(transfers):,}")
    print(f"  rows after: {len(filtered):,}")
    print(f"  rows removed: {len(transfers) - len(filtered):,}")
    print(f"  unique player_ids in filtered players: {len(valid_player_ids):,}")
    print(f"  unique player_ids in transfers before: {transfer_player_ids.nunique(dropna=True):,}")
    print(f"  example removed player_ids: {unmatched_ids}")

    return normalize_integer_like_columns(
        filtered,
        ["transfer_id", "player_id", "from_club_id", "to_club_id"],
    )


def build_transfers(
    input_dir: Path, output_dir: Path, players: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Load, clean, optionally player-filter, and save transfers.csv."""
    raw_transfers = load_csv(input_dir, "transfers.csv")
    transfers = clean_transfers(raw_transfers)
    if players is not None:
        transfers = filter_transfers_to_players(transfers, players)
    save_table(transfers, output_dir, "transfers.csv")
    return transfers


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
    build_clubs(input_dir, output_dir)
    build_countries(input_dir, output_dir)
    build_transfers(input_dir, output_dir, players)

    print("\nDone.")


if __name__ == "__main__":
    main()
