"""Analyze benchmark metadata for cleaned Transfermarkt/player-scores tables.

The script reads processed CSV files from ``dataset_clean/`` by default, writes a
human-readable report to ``dataset_clean/metadata_report.txt``, and saves candidate
entity files under ``dataset_clean/metadata/``.
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path
from textwrap import indent
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("dataset_clean")
PRIMARY_START = pd.Timestamp("2023-01-01")
PRIMARY_END = pd.Timestamp("2025-12-31")
WIDE_START = pd.Timestamp("2020-01-01")
WIDE_END = pd.Timestamp("2025-12-31")

TABLE_FILES = {
    "players": "players.csv",
    "clubs": "clubs.csv",
    "countries": "countries.csv",
    "transfers": "transfers.csv",
}

ID_COLUMNS = {
    "players": "player_id",
    "clubs": "club_id",
    "countries": "country_id",
    "transfers": "transfer_id",
}

CANDIDATE_SCHEMAS = {
    "candidate_capital_questions.csv": [
        "country_id",
        "country_name",
        "capital_city",
        "confederation",
        "continent",
    ],
    "candidate_birth_year_questions.csv": [
        "player_id",
        "name",
        "date_of_birth",
        "birth_year",
    ],
    "candidate_age_questions.csv": [
        "player_id",
        "name",
        "date_of_birth",
        "age_on_2025_12_31",
    ],
    "candidate_birth_city_questions.csv": [
        "player_id",
        "name",
        "city_of_birth",
        "country_of_birth",
    ],
    "candidate_club_country_questions.csv": [
        "club_id",
        "name",
        "country_name",
        "country_source",
    ],
    "candidate_transfer_path_questions.csv": [
        "player_id",
        "player_name",
        "path_step",
        "transfer_date",
        "from_club_name",
        "to_club_name",
        "from_club_id",
        "to_club_id",
        "is_continuous_path",
    ],
    "candidate_same_club_questions.csv": [
        "club_id",
        "club_name",
        "player_1_id",
        "player_1_name",
        "player_1_join_date",
        "player_1_leave_date",
        "player_2_id",
        "player_2_name",
        "player_2_join_date",
        "player_2_leave_date",
        "overlap_start_date",
        "overlap_end_date",
        "overlap_days",
        "gap_start_date",
        "gap_end_date",
        "gap_days",
    ],
    "candidate_same_club_non_overlap_questions.csv": [
        "club_id",
        "club_name",
        "player_1_id",
        "player_1_name",
        "player_1_join_date",
        "player_1_leave_date",
        "player_2_id",
        "player_2_name",
        "player_2_join_date",
        "player_2_leave_date",
        "overlap_start_date",
        "overlap_end_date",
        "overlap_days",
        "gap_start_date",
        "gap_end_date",
        "gap_days",
    ],
    "candidate_citizenship_club_country_questions.csv": [
        "player_id",
        "player_name",
        "citizenship_country",
        "club_id",
        "club_name",
        "club_country",
        "club_country_source",
        "evidence_transfer_date",
    ],
    "candidate_club_confederation_questions.csv": [
        "club_id",
        "name",
        "country_name",
        "confederation",
        "continent",
        "country_source",
    ],
}


class Report:
    """Collect report lines and print selected high-level findings."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.console_lines: list[str] = []

    def section(self, title: str) -> None:
        self.lines.append("")
        self.lines.append("=" * len(title))
        self.lines.append(title)
        self.lines.append("=" * len(title))

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def warn(self, message: str) -> None:
        line = f"WARNING: {message}"
        self.lines.append(line)
        print(line)

    def important(self, message: str) -> None:
        self.console_lines.append(message)
        print(message)
        self.lines.append(message)

    def add_frame(
        self,
        title: str,
        frame: pd.DataFrame | pd.Series | None,
        max_rows: int | None = 20,
    ) -> None:
        self.lines.append(title)
        if frame is None:
            self.lines.append("  (not available)")
            return

        if isinstance(frame, pd.Series):
            frame = frame.reset_index()

        if frame.empty:
            self.lines.append("  (none)")
            return

        display = frame if max_rows is None else frame.head(max_rows)
        self.lines.append(indent(display.to_string(index=False), "  "))
        if max_rows is not None and len(frame) > max_rows:
            self.lines.append(f"  ... {len(frame) - max_rows:,} more rows omitted")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines).strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze metadata for cleaned Transfermarkt benchmark tables."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing processed CSV files. Defaults to dataset_clean/.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Path for the text report. Defaults to <input-dir>/metadata_report.txt.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help="Directory for candidate CSV files. Defaults to <input-dir>/metadata/.",
    )
    parser.add_argument(
        "--max-same-club-pairs",
        type=int,
        default=500,
        help="Maximum same-club player pairs to save, to avoid quadratic blowups.",
    )
    return parser.parse_args()


def load_csv(path: Path, report: Report, required: bool = True) -> pd.DataFrame | None:
    """Load a CSV file if available."""
    if not path.exists():
        if required:
            report.warn(f"Required file is missing: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive CLI behavior.
        report.warn(f"Could not load {path}: {exc}")
        return None

    report.add(f"Loaded {path}: {len(df):,} rows, {len(df.columns):,} columns")
    return df


def load_tables(input_dir: Path, report: Report) -> dict[str, pd.DataFrame | None]:
    """Load the four expected benchmark tables and an optional competitions table."""
    tables: dict[str, pd.DataFrame | None] = {}
    for table_name, filename in TABLE_FILES.items():
        tables[table_name] = load_csv(input_dir / filename, report, required=True)

    # The cleaned benchmark does not require competitions.csv, but if the user
    # later adds it to dataset_clean/ we can use it to infer club countries.
    competitions = load_csv(input_dir / "competitions.csv", report, required=False)
    if competitions is not None:
        tables["competitions"] = competitions
    return tables


def has_columns(
    df: pd.DataFrame | None,
    columns: list[str],
    table_name: str,
    report: Report,
    analysis_name: str,
) -> bool:
    """Return False and warn instead of crashing when columns are missing."""
    if df is None:
        report.warn(f"Skipping {analysis_name}: {table_name}.csv is not available.")
        return False

    missing = [column for column in columns if column not in df.columns]
    if missing:
        report.warn(
            f"Skipping {analysis_name}: {table_name}.csv is missing columns {missing}."
        )
        return False
    return True


def valid_mask(series: pd.Series) -> pd.Series:
    """Treat NaN and blank strings as missing."""
    text = series.astype("string").str.strip()
    return (series.notna() & text.notna() & text.ne("")).fillna(False)


def valid_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(valid_mask(df[column]).sum())


def normalize_key_value(value: Any) -> str | None:
    """Normalize IDs for join checks while preserving numeric identity."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return str(int(numeric)) if float(numeric).is_integer() else str(numeric)
    return text


def normalize_key_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_key_value).astype("string")


def normalize_name(value: Any) -> str:
    """Normalize country-like names for style comparison."""
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def value_counts_frame(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[label, "count"])

    counts = df.loc[valid_mask(df[column]), column].value_counts(dropna=True)
    return counts.rename_axis(label).reset_index(name="count")


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def format_date_series(series: pd.Series) -> pd.Series:
    parsed = parse_dates(series)
    return parsed.dt.strftime("%Y-%m-%d").astype("string")


def first_valid(series: pd.Series) -> Any:
    mask = valid_mask(series)
    if mask.any():
        return series.loc[mask].iloc[0]
    return pd.NA


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def ensure_columns(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    """Return a frame with the requested columns, adding missing ones if needed."""
    if df is None:
        return pd.DataFrame(columns=columns)

    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[columns]


def analyze_basic_table_stats(
    tables: dict[str, pd.DataFrame | None],
    report: Report,
) -> None:
    report.section("1. Basic Table Statistics")
    for table_name in TABLE_FILES:
        df = tables.get(table_name)
        report.add("")
        report.add(f"{table_name}.csv")
        report.add("-" * (len(table_name) + 4))

        if df is None:
            report.add("Table not available.")
            continue

        report.add(f"Rows: {len(df):,}")
        report.add(f"Columns: {len(df.columns):,}")
        report.add(f"Column names: {list(df.columns)}")
        report.add(f"Duplicated rows: {int(df.duplicated().sum()):,}")

        missing = pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": [
                    int((~valid_mask(df[column])).sum()) for column in df.columns
                ],
                "unique_values": [
                    int(df.loc[valid_mask(df[column]), column].nunique(dropna=True))
                    for column in df.columns
                ],
            }
        )
        report.add_frame("Missing and unique counts:", missing, max_rows=None)

        id_column = ID_COLUMNS.get(table_name)
        if id_column in df.columns:
            ids = normalize_key_series(df[id_column])
            valid_ids = ids[ids.notna()]
            duplicate_id_rows = int(valid_ids.duplicated(keep=False).sum())
            duplicate_id_values = int(valid_ids[valid_ids.duplicated()].nunique())
            report.add(f"ID column: {id_column}")
            report.add(f"Rows with duplicated IDs: {duplicate_id_rows:,}")
            report.add(f"Duplicated ID values: {duplicate_id_values:,}")
        else:
            report.add(f"ID column not found: {id_column}")


def unique_name_preferred_players(players: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Prefer unique player names for cleaner benchmark candidates."""
    subset = players.loc[mask].copy()
    if "name" not in subset.columns:
        return subset

    counts = players.loc[valid_mask(players["name"]), "name"].value_counts()
    subset["_name_count"] = subset["name"].map(counts)
    unique_subset = subset[subset["_name_count"].eq(1)]
    if len(unique_subset) >= 20:
        subset = unique_subset
    return subset.drop(columns=["_name_count"], errors="ignore")


def add_birth_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dob = parse_dates(out["date_of_birth"])
    out["date_of_birth"] = dob.dt.strftime("%Y-%m-%d").astype("string")
    out["birth_year"] = dob.dt.year.astype("Int64")
    return out


def compute_age_on_2025_12_31(dob: pd.Series) -> pd.Series:
    target = pd.Timestamp("2025-12-31")
    age = target.year - dob.dt.year
    birthday_not_reached = (dob.dt.month > target.month) | (
        (dob.dt.month == target.month) & (dob.dt.day > target.day)
    )
    return (age - birthday_not_reached.astype("Int64")).astype("Int64")


def analyze_players(
    players: pd.DataFrame | None,
    report: Report,
) -> dict[str, pd.DataFrame]:
    report.section("2. Player Metadata")
    candidates: dict[str, pd.DataFrame] = {}

    if players is None:
        report.warn("Skipping player metadata: players.csv is not available.")
        return candidates

    report.important(f"Players: {len(players):,}")
    if "last_season" in players.columns:
        last_season = pd.to_numeric(players["last_season"], errors="coerce")
        distribution = (
            last_season.value_counts()
            .sort_index()
            .rename_axis("last_season")
            .reset_index(name="player_count")
        )
        report.add_frame("Distribution of last_season:", distribution, max_rows=None)
        in_window = int(last_season.between(2023, 2026, inclusive="both").sum())
        report.important(f"Players with 2023 <= last_season <= 2026: {in_window:,}")
    else:
        report.warn("players.csv has no last_season column.")

    for column in [
        "date_of_birth",
        "country_of_birth",
        "city_of_birth",
        "country_of_citizenship",
        "position",
        "height",
    ]:
        if column == "date_of_birth" and column in players.columns:
            count = int(parse_dates(players[column]).notna().sum())
        else:
            count = valid_count(players, column)
        report.add(f"Players with valid {column}: {count:,}")

    report.add_frame(
        "Top 20 countries of birth by player count:",
        value_counts_frame(players, "country_of_birth", "country_of_birth"),
    )
    report.add_frame(
        "Top 20 countries of citizenship by player count:",
        value_counts_frame(players, "country_of_citizenship", "country_of_citizenship"),
    )
    report.add_frame(
        "Top 20 birth cities by player count:",
        value_counts_frame(players, "city_of_birth", "city_of_birth"),
    )

    if "name" in players.columns:
        name_counts = players.loc[valid_mask(players["name"]), "name"].value_counts()
        duplicated_names = name_counts[name_counts > 1]
        report.add(f"Duplicated player names: {len(duplicated_names):,}")
        report.add_frame(
            "Examples of duplicated player names:",
            duplicated_names.rename_axis("name").reset_index(name="count"),
        )

    if has_columns(
        players,
        ["player_id", "name", "date_of_birth"],
        "players",
        report,
        "birth-year and age candidates",
    ):
        dob = parse_dates(players["date_of_birth"])
        valid_dob = dob.notna() & valid_mask(players["name"])
        birth_candidates = unique_name_preferred_players(players, valid_dob)
        birth_candidates = add_birth_columns(birth_candidates)
        birth_candidates = birth_candidates[[
            "player_id",
            "name",
            "date_of_birth",
            "birth_year",
        ]].head(20)
        candidates["candidate_birth_year_questions.csv"] = birth_candidates

        age_candidates = unique_name_preferred_players(players, valid_dob)
        age_dob = parse_dates(age_candidates["date_of_birth"])
        age_candidates = age_candidates.copy()
        age_candidates["date_of_birth"] = age_dob.dt.strftime("%Y-%m-%d").astype("string")
        age_candidates["age_on_2025_12_31"] = compute_age_on_2025_12_31(age_dob)
        age_candidates = age_candidates[[
            "player_id",
            "name",
            "date_of_birth",
            "age_on_2025_12_31",
        ]].head(20)
        candidates["candidate_age_questions.csv"] = age_candidates

    if has_columns(
        players,
        ["player_id", "name", "city_of_birth", "country_of_birth"],
        "players",
        report,
        "birth-city candidates",
    ):
        birth_city_mask = (
            valid_mask(players["name"])
            & valid_mask(players["city_of_birth"])
            & valid_mask(players["country_of_birth"])
        )
        birth_city_candidates = unique_name_preferred_players(players, birth_city_mask)
        birth_city_candidates = birth_city_candidates[[
            "player_id",
            "name",
            "city_of_birth",
            "country_of_birth",
        ]].head(20)
        candidates["candidate_birth_city_questions.csv"] = birth_city_candidates

    return candidates


def analyze_country_style(
    countries: pd.DataFrame | None,
    players: pd.DataFrame | None,
    clubs: pd.DataFrame | None,
    report: Report,
) -> None:
    """Compare country names used across tables when country columns exist."""
    if countries is None or "country_name" not in countries.columns:
        report.warn("Skipping country naming-style checks: countries.country_name missing.")
        return

    country_names = set(countries.loc[valid_mask(countries["country_name"]), "country_name"])
    normalized_country_names = {normalize_name(name) for name in country_names}

    if players is not None:
        for column in ["country_of_birth", "country_of_citizenship"]:
            if column not in players.columns:
                report.warn(f"players.csv has no {column} column for country style checks.")
                continue

            values = set(players.loc[valid_mask(players[column]), column])
            exact_unmatched = sorted(values - country_names)
            normalized_unmatched = sorted(
                value for value in values if normalize_name(value) not in normalized_country_names
            )
            report.add_frame(
                f"Player {column} values not found exactly in countries.country_name:",
                pd.DataFrame({column: exact_unmatched}),
            )
            report.add_frame(
                f"Player {column} values not found after normalized comparison:",
                pd.DataFrame({column: normalized_unmatched}),
            )

    if clubs is None:
        report.warn("Skipping club country naming-style checks: clubs.csv is not available.")
        return

    club_country_columns = [
        column for column in clubs.columns if "country" in column.lower()
    ]
    if not club_country_columns:
        report.add(
            "clubs.csv has no country-like column, so club country naming style "
            "cannot be compared directly."
        )
        return

    for column in club_country_columns:
        values = set(clubs.loc[valid_mask(clubs[column]), column])
        exact_unmatched = sorted(values - country_names)
        report.add_frame(
            f"Club {column} values not found exactly in countries.country_name:",
            pd.DataFrame({column: exact_unmatched}),
        )


def analyze_countries(
    countries: pd.DataFrame | None,
    players: pd.DataFrame | None,
    clubs: pd.DataFrame | None,
    report: Report,
) -> dict[str, pd.DataFrame]:
    report.section("3. Country Metadata")
    candidates: dict[str, pd.DataFrame] = {}

    if countries is None:
        report.warn("Skipping country metadata: countries.csv is not available.")
        return candidates

    report.important(f"Countries: {len(countries):,}")
    for column in ["capital_city", "confederation"]:
        report.add(f"Countries with valid {column}: {valid_count(countries, column):,}")

    for column, title in [
        ("country_name", "Duplicated country names:"),
        ("capital_city", "Duplicated capital cities:"),
    ]:
        if column in countries.columns:
            counts = countries.loc[valid_mask(countries[column]), column].value_counts()
            duplicated = counts[counts > 1].rename_axis(column).reset_index(name="count")
            report.add(f"{title} {len(duplicated):,}")
            report.add_frame(f"Examples for {title.lower()}", duplicated)

    if "capital_city" in countries.columns:
        missing_capitals = countries.loc[~valid_mask(countries["capital_city"])]
        report.add_frame(
            "Countries missing capital city:",
            missing_capitals[[column for column in ["country_id", "country_name"] if column in countries.columns]],
        )
    if "confederation" in countries.columns:
        missing_confederation = countries.loc[~valid_mask(countries["confederation"])]
        report.add_frame(
            "Countries missing confederation:",
            missing_confederation[[column for column in ["country_id", "country_name"] if column in countries.columns]],
        )

    report.add_frame(
        "Top confederations by country count:",
        value_counts_frame(countries, "confederation", "confederation"),
        max_rows=None,
    )

    analyze_country_style(countries, players, clubs, report)

    if has_columns(
        countries,
        ["country_id", "country_name", "capital_city", "confederation", "continent"],
        "countries",
        report,
        "capital-city candidates",
    ):
        capital_counts = countries.loc[
            valid_mask(countries["capital_city"]), "capital_city"
        ].value_counts()
        good_capital = countries["capital_city"].map(capital_counts).eq(1)
        candidate_mask = (
            valid_mask(countries["country_name"])
            & valid_mask(countries["capital_city"])
            & good_capital.fillna(False)
        )
        capital_candidates = countries.loc[candidate_mask, [
            "country_id",
            "country_name",
            "capital_city",
            "confederation",
            "continent",
        ]].head(20)
        candidates["candidate_capital_questions.csv"] = capital_candidates

    return candidates


def country_enriched_club_map(
    club_map: pd.DataFrame,
    countries: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach country metadata to a club-country mapping."""
    if club_map.empty or countries is None or "country_name" not in countries.columns:
        return club_map

    countries_lookup = countries.copy()
    countries_lookup["_country_key"] = countries_lookup["country_name"].map(normalize_name)

    enriched = club_map.copy()
    enriched["_country_key"] = enriched["country_name"].map(normalize_name)
    enriched = enriched.merge(
        countries_lookup[[
            "_country_key",
            *[column for column in ["confederation", "continent"] if column in countries_lookup.columns],
        ]].drop_duplicates("_country_key"),
        on="_country_key",
        how="left",
    )
    return enriched.drop(columns=["_country_key"], errors="ignore")


def infer_club_country_map(
    clubs: pd.DataFrame | None,
    countries: pd.DataFrame | None,
    competitions: pd.DataFrame | None,
    report: Report,
) -> pd.DataFrame:
    """Infer club countries directly or via an optional competitions table."""
    output_columns = [
        "club_id",
        "name",
        "country_name",
        "country_source",
        "confederation",
        "continent",
    ]
    if clubs is None:
        return pd.DataFrame(columns=output_columns)

    direct_country_columns = [
        "country",
        "country_name",
        "club_country",
        "country_of_club",
    ]
    country_column = first_existing_column(clubs, direct_country_columns)

    if country_column is not None:
        base_columns = [
            column for column in ["club_id", "name", country_column] if column in clubs.columns
        ]
        club_map = clubs[base_columns].copy()
        club_map = club_map.rename(columns={country_column: "country_name"})
        club_map["country_source"] = f"clubs.{country_column}"
        report.add(f"Club country source: direct column clubs.{country_column}")
        return ensure_columns(country_enriched_club_map(club_map, countries), output_columns)

    if "domestic_competition_id" not in clubs.columns:
        report.add(
            "Club country mapping is not available: clubs.csv has no country "
            "column and no domestic_competition_id column."
        )
        return pd.DataFrame(columns=output_columns)

    if competitions is None:
        report.add(
            "Club country mapping is not available from the processed tables: "
            "clubs.csv has domestic_competition_id, but a competition-to-country "
            "table such as competitions.csv is not present in dataset_clean/."
        )
        return pd.DataFrame(columns=output_columns)

    competition_id_column = first_existing_column(
        competitions,
        ["competition_id", "domestic_competition_id", "competition_code"],
    )
    competition_country_column = first_existing_column(
        competitions,
        ["country_name", "country", "competition_country"],
    )
    if competition_id_column is None or competition_country_column is None:
        report.add(
            "Club country mapping is not available: competitions.csv exists, "
            "but it does not contain a recognizable competition ID and country column."
        )
        return pd.DataFrame(columns=output_columns)

    clubs_work = clubs.copy()
    competitions_work = competitions[[competition_id_column, competition_country_column]].copy()
    clubs_work["_competition_key"] = normalize_key_series(clubs_work["domestic_competition_id"])
    competitions_work["_competition_key"] = normalize_key_series(
        competitions_work[competition_id_column]
    )
    merged = clubs_work.merge(
        competitions_work[["_competition_key", competition_country_column]].drop_duplicates(
            "_competition_key"
        ),
        on="_competition_key",
        how="left",
    )
    club_map = merged[[
        *[column for column in ["club_id", "name"] if column in merged.columns],
        competition_country_column,
    ]].rename(columns={competition_country_column: "country_name"})
    club_map["country_source"] = (
        f"competitions.{competition_country_column} via clubs.domestic_competition_id"
    )
    report.add(
        "Club country source: competitions.csv joined through domestic_competition_id."
    )
    return ensure_columns(country_enriched_club_map(club_map, countries), output_columns)


def analyze_clubs(
    clubs: pd.DataFrame | None,
    countries: pd.DataFrame | None,
    competitions: pd.DataFrame | None,
    report: Report,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    report.section("4. Club Metadata")
    candidates: dict[str, pd.DataFrame] = {}

    if clubs is None:
        report.warn("Skipping club metadata: clubs.csv is not available.")
        return candidates, pd.DataFrame()

    report.important(f"Clubs: {len(clubs):,}")
    for column in ["domestic_competition_id", "stadium_name", "stadium_seats"]:
        report.add(f"Clubs with valid {column}: {valid_count(clubs, column):,}")

    if "name" in clubs.columns:
        name_counts = clubs.loc[valid_mask(clubs["name"]), "name"].value_counts()
        duplicated_names = name_counts[name_counts > 1]
        report.add(f"Duplicated club names: {len(duplicated_names):,}")
        report.add_frame(
            "Examples of duplicated club names:",
            duplicated_names.rename_axis("name").reset_index(name="count"),
        )

    report.add_frame(
        "Top domestic competitions by club count:",
        value_counts_frame(clubs, "domestic_competition_id", "domestic_competition_id"),
    )

    club_country_map = infer_club_country_map(clubs, countries, competitions, report)
    mapped_count = int(valid_mask(club_country_map["country_name"]).sum()) if not club_country_map.empty else 0
    report.important(f"Clubs mapped to a country: {mapped_count:,}")

    if mapped_count:
        club_country_candidates = club_country_map.loc[
            valid_mask(club_country_map["country_name"]),
            ["club_id", "name", "country_name", "country_source"],
        ].head(20)
        candidates["candidate_club_country_questions.csv"] = club_country_candidates

        if {"confederation", "continent"}.issubset(club_country_map.columns):
            confed_candidates = club_country_map.loc[
                valid_mask(club_country_map["country_name"])
                & valid_mask(club_country_map["confederation"]),
                [
                    "club_id",
                    "name",
                    "country_name",
                    "confederation",
                    "continent",
                    "country_source",
                ],
            ].head(20)
            candidates["candidate_club_confederation_questions.csv"] = confed_candidates

    return candidates, club_country_map


def prepare_transfers(transfers: pd.DataFrame | None) -> pd.DataFrame | None:
    if transfers is None:
        return None

    out = transfers.copy()
    if "transfer_date" in out.columns:
        out["_transfer_date_dt"] = parse_dates(out["transfer_date"])
    return out


def date_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if "_transfer_date_dt" not in df.columns:
        return df.iloc[0:0].copy()
    return df.loc[df["_transfer_date_dt"].between(start, end, inclusive="both")].copy()


def top_transfer_players(transfers: pd.DataFrame) -> pd.DataFrame:
    group_columns = [column for column in ["player_id", "player_name"] if column in transfers.columns]
    if not group_columns:
        return pd.DataFrame(columns=["transfer_count"])
    return (
        transfers.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="transfer_count")
        .sort_values("transfer_count", ascending=False)
    )


def top_transfer_clubs(
    transfers: pd.DataFrame,
    id_column: str,
    name_column: str,
    count_column: str,
) -> pd.DataFrame:
    group_columns = [column for column in [id_column, name_column] if column in transfers.columns]
    if not group_columns:
        return pd.DataFrame(columns=[count_column])
    return (
        transfers.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name=count_column)
        .sort_values(count_column, ascending=False)
    )


def analyze_transfers(
    transfers: pd.DataFrame | None,
    report: Report,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    report.section("5. Transfer Metadata")
    candidates: dict[str, pd.DataFrame] = {}

    transfers_work = prepare_transfers(transfers)
    if transfers_work is None:
        report.warn("Skipping transfer metadata: transfers.csv is not available.")
        return candidates, None

    report.important(f"Transfers: {len(transfers_work):,}")
    if "_transfer_date_dt" in transfers_work.columns:
        valid_dates = transfers_work["_transfer_date_dt"].dropna()
        if valid_dates.empty:
            report.add("transfer_date range: no valid dates")
        else:
            report.important(
                "Transfer date range: "
                f"{valid_dates.min().date()} to {valid_dates.max().date()}"
            )
        primary_window = date_window(transfers_work, PRIMARY_START, PRIMARY_END)
        report.important(
            "Transfers between 2023-01-01 and 2025-12-31: "
            f"{len(primary_window):,}"
        )
    else:
        report.warn("transfers.csv has no transfer_date column.")
        primary_window = transfers_work.iloc[0:0].copy()

    for column, label in [
        ("player_id", "unique players in transfer records"),
        ("from_club_id", "unique from_club_id"),
        ("to_club_id", "unique to_club_id"),
    ]:
        if column in transfers_work.columns:
            report.add(f"Number of {label}: {transfers_work[column].nunique(dropna=True):,}")
        else:
            report.warn(f"transfers.csv has no {column} column.")

    for column in [
        "player_id",
        "from_club_id",
        "to_club_id",
        "from_club_name",
        "to_club_name",
    ]:
        report.add(f"Transfers with missing {column}: {len(transfers_work) - valid_count(transfers_work, column):,}")

    report.add_frame(
        "Top 20 players by number of transfers:",
        top_transfer_players(transfers_work),
    )
    report.add_frame(
        "Top 20 clubs by number of incoming transfers:",
        top_transfer_clubs(
            transfers_work, "to_club_id", "to_club_name", "incoming_transfer_count"
        ),
    )
    report.add_frame(
        "Top 20 clubs by number of outgoing transfers:",
        top_transfer_clubs(
            transfers_work, "from_club_id", "from_club_name", "outgoing_transfer_count"
        ),
    )

    return candidates, transfers_work


def transfer_sort_columns(transfers: pd.DataFrame) -> list[str]:
    columns = ["_transfer_date_dt"]
    if "transfer_id" in transfers.columns:
        columns.append("transfer_id")
    return [column for column in columns if column in transfers.columns]


def continuous_check(group: pd.DataFrame) -> bool:
    """Check whether consecutive transfer IDs connect when both IDs are known."""
    rows = group.to_dict("records")
    for previous, current in zip(rows, rows[1:]):
        previous_to = normalize_key_value(previous.get("to_club_id"))
        current_from = normalize_key_value(current.get("from_club_id"))
        if previous_to is not None and current_from is not None and previous_to != current_from:
            return False
    return True


def build_transfer_path_rows(
    transfers: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[int, pd.DataFrame]:
    """Return ordered path rows for players with at least two transfers."""
    output_columns = CANDIDATE_SCHEMAS["candidate_transfer_path_questions.csv"]
    if transfers is None:
        return 0, pd.DataFrame(columns=output_columns)

    required = [
        "player_id",
        "player_name",
        "transfer_date",
        "from_club_name",
        "to_club_name",
        "from_club_id",
        "to_club_id",
        "_transfer_date_dt",
    ]
    if any(column not in transfers.columns for column in required):
        return 0, pd.DataFrame(columns=output_columns)

    window = date_window(transfers, start, end)
    if window.empty:
        return 0, pd.DataFrame(columns=output_columns)

    rows: list[dict[str, Any]] = []
    players_with_at_least_two = 0
    for player_id, group in window.groupby("player_id", dropna=True):
        if len(group) < 2:
            continue

        players_with_at_least_two += 1
        ordered = group.sort_values(transfer_sort_columns(group), kind="mergesort")
        is_continuous = continuous_check(ordered)
        for step, (_, transfer) in enumerate(ordered.iterrows(), start=1):
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": transfer.get("player_name"),
                    "path_step": step,
                    "transfer_date": transfer["_transfer_date_dt"].strftime("%Y-%m-%d")
                    if pd.notna(transfer["_transfer_date_dt"])
                    else pd.NA,
                    "from_club_name": transfer.get("from_club_name"),
                    "to_club_name": transfer.get("to_club_name"),
                    "from_club_id": transfer.get("from_club_id"),
                    "to_club_id": transfer.get("to_club_id"),
                    "is_continuous_path": is_continuous,
                }
            )

    return players_with_at_least_two, pd.DataFrame(rows, columns=output_columns)


def analyze_continuous_transfers(
    transfers: pd.DataFrame | None,
    report: Report,
) -> dict[str, pd.DataFrame]:
    report.section("6. Continuous Transfer Analysis")
    candidates: dict[str, pd.DataFrame] = {}

    players_2_primary, primary_paths = build_transfer_path_rows(
        transfers, PRIMARY_START, PRIMARY_END
    )
    continuous_primary = primary_paths.loc[
        primary_paths["is_continuous_path"].eq(True)
    ].copy()
    continuous_player_count = (
        continuous_primary["player_id"].nunique(dropna=True)
        if not continuous_primary.empty
        else 0
    )

    report.important(
        "Players with at least 2 transfers in 2023-2025: "
        f"{players_2_primary:,}"
    )
    report.important(
        "Players with continuous transfer paths in 2023-2025: "
        f"{continuous_player_count:,}"
    )

    names = (
        sorted(continuous_primary["player_name"].dropna().astype(str).unique())
        if not continuous_primary.empty
        else []
    )
    report.add_frame(
        "Names of players with continuous transfer paths in 2023-2025:",
        pd.DataFrame({"player_name": names}),
        max_rows=None,
    )
    report.add_frame(
        "Ordered continuous transfer paths in 2023-2025:",
        continuous_primary,
        max_rows=None,
    )

    candidates["candidate_transfer_path_questions.csv"] = primary_paths

    if continuous_player_count < 5:
        players_2_wide, wide_paths = build_transfer_path_rows(
            transfers, WIDE_START, WIDE_END
        )
        continuous_wide = wide_paths.loc[wide_paths["is_continuous_path"].eq(True)].copy()
        continuous_wide_player_count = (
            continuous_wide["player_id"].nunique(dropna=True)
            if not continuous_wide.empty
            else 0
        )
        report.add("")
        report.add(
            "Very few continuous paths were found for 2023-2025, so the wider "
            "2020-2025 window was also checked."
        )
        report.add(
            f"Players with at least 2 transfers in 2020-2025: {players_2_wide:,}"
        )
        report.add(
            "Players with continuous transfer paths in 2020-2025: "
            f"{continuous_wide_player_count:,}"
        )
        report.add_frame(
            "Ordered continuous transfer paths in 2020-2025:",
            continuous_wide,
            max_rows=None if len(continuous_wide) <= 300 else 300,
        )

    return candidates


def analyze_club_activity(
    transfers: pd.DataFrame | None,
    report: Report,
    max_same_club_pairs: int,
) -> dict[str, pd.DataFrame]:
    report.section("7. Club Transfer Activity Analysis")
    candidates: dict[str, pd.DataFrame] = {}

    if transfers is None or "_transfer_date_dt" not in transfers.columns:
        report.warn("Skipping club transfer activity: transfer dates are not available.")
        return candidates

    window = date_window(transfers, PRIMARY_START, PRIMARY_END)
    if window.empty:
        report.warn("Skipping club transfer activity: no transfers in 2023-2025.")
        return candidates

    incoming = pd.DataFrame()
    if {"to_club_id", "to_club_name", "player_id"}.issubset(window.columns):
        incoming = (
            window.groupby("to_club_id", dropna=False)
            .agg(
                club_name=("to_club_name", first_valid),
                incoming_transfers=("player_id", "size"),
                unique_incoming_players=("player_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"to_club_id": "club_id"})
        )

    outgoing = pd.DataFrame()
    if {"from_club_id", "from_club_name", "player_id"}.issubset(window.columns):
        outgoing = (
            window.groupby("from_club_id", dropna=False)
            .agg(
                club_name=("from_club_name", first_valid),
                outgoing_transfers=("player_id", "size"),
                unique_outgoing_players=("player_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"from_club_id": "club_id"})
        )

    if incoming.empty and outgoing.empty:
        report.warn("Skipping club transfer activity: club ID columns are missing.")
        return candidates

    if not incoming.empty and not outgoing.empty:
        activity = incoming.merge(
            outgoing,
            on="club_id",
            how="outer",
            suffixes=("_incoming", "_outgoing"),
        )
    elif not incoming.empty:
        activity = incoming.copy()
    else:
        activity = outgoing.copy()

    if "club_name_incoming" in activity.columns or "club_name_outgoing" in activity.columns:
        incoming_names = activity.get("club_name_incoming")
        outgoing_names = activity.get("club_name_outgoing")
        if incoming_names is not None and outgoing_names is not None:
            activity["club_name"] = incoming_names.combine_first(outgoing_names)
        elif incoming_names is not None:
            activity["club_name"] = incoming_names
        else:
            activity["club_name"] = outgoing_names
        activity = activity.drop(
            columns=["club_name_incoming", "club_name_outgoing"], errors="ignore"
        )

    for column in [
        "incoming_transfers",
        "unique_incoming_players",
        "outgoing_transfers",
        "unique_outgoing_players",
    ]:
        if column not in activity.columns:
            activity[column] = 0
        activity[column] = activity[column].fillna(0).astype("Int64")

    report.add_frame(
        "Top 20 clubs by incoming transfers in 2023-2025:",
        activity.sort_values("incoming_transfers", ascending=False),
    )
    report.add_frame(
        "Top 20 clubs by outgoing transfers in 2023-2025:",
        activity.sort_values("outgoing_transfers", ascending=False),
    )
    both = activity.loc[
        activity["incoming_transfers"].gt(0) & activity["outgoing_transfers"].gt(0)
    ].copy()
    report.important(f"Clubs with both incoming and outgoing transfers: {len(both):,}")
    report.add_frame(
        "Clubs with both incoming and outgoing transfers:",
        both.sort_values(["incoming_transfers", "outgoing_transfers"], ascending=False),
    )

    same_club_candidates, non_overlap_candidates, spell_count = build_same_club_candidates(
        window, max_same_club_pairs
    )
    report.add(f"Complete player-club spells inferred from transfer rows: {spell_count:,}")
    report.add_frame(
        "Candidate same-club teammate pairs with overlapping club spells:",
        same_club_candidates,
    )
    report.add_frame(
        "Candidate same-club pairs with no overlapping club spell:",
        non_overlap_candidates,
    )
    if len(same_club_candidates) >= max_same_club_pairs:
        report.add(
            f"Same-club teammate candidate generation stopped at {max_same_club_pairs:,} pairs."
        )
    if len(non_overlap_candidates) >= max_same_club_pairs:
        report.add(
            f"Same-club non-overlap candidate generation stopped at {max_same_club_pairs:,} pairs."
        )
    candidates["candidate_same_club_questions.csv"] = same_club_candidates
    candidates["candidate_same_club_non_overlap_questions.csv"] = non_overlap_candidates
    return candidates


def is_real_team(club_id: Any, club_name: Any) -> bool:
    """Exclude placeholder destinations that are not actual teams."""
    club_key = normalize_key_value(club_id)
    if club_key is None:
        return False

    if pd.isna(club_name):
        return True

    normalized_name = normalize_name(club_name)
    non_team_names = {
        "without club",
        "retired",
        "career break",
        "unknown",
        "not assigned",
    }
    return normalized_name not in non_team_names


def build_complete_club_spells(transfers: pd.DataFrame) -> pd.DataFrame:
    """Infer player-club spells with known transfer-in and transfer-out dates.

    A complete spell is created when a player transfers into a club and later has
    a transfer row leaving that same club. Open-ended spells are intentionally
    skipped because they cannot prove overlap or non-overlap with another player.
    """
    output_columns = [
        "club_id",
        "club_name",
        "player_id",
        "player_name",
        "join_date",
        "leave_date",
    ]
    required = {
        "player_id",
        "player_name",
        "_transfer_date_dt",
        "from_club_id",
        "from_club_name",
        "to_club_id",
        "to_club_name",
    }
    if not required.issubset(transfers.columns):
        return pd.DataFrame(columns=output_columns)

    rows: list[dict[str, Any]] = []
    for player_id, group in transfers.groupby("player_id", dropna=True):
        ordered = group.sort_values(transfer_sort_columns(group), kind="mergesort")
        records = ordered.to_dict("records")
        for index, incoming in enumerate(records):
            join_date = incoming.get("_transfer_date_dt")
            to_club_id = incoming.get("to_club_id")
            to_club_name = incoming.get("to_club_name")
            to_club_key = normalize_key_value(to_club_id)

            if pd.isna(join_date) or not is_real_team(to_club_id, to_club_name):
                continue

            for outgoing in records[index + 1 :]:
                leave_date = outgoing.get("_transfer_date_dt")
                if pd.isna(leave_date) or leave_date <= join_date:
                    continue

                from_club_key = normalize_key_value(outgoing.get("from_club_id"))
                if from_club_key != to_club_key:
                    continue

                rows.append(
                    {
                        "club_id": to_club_id,
                        "club_name": first_valid(
                            pd.Series([to_club_name, outgoing.get("from_club_name")])
                        ),
                        "player_id": player_id,
                        "player_name": incoming.get("player_name"),
                        "join_date": join_date,
                        "leave_date": leave_date,
                    }
                )
                break

    return pd.DataFrame(rows, columns=output_columns)


def format_pair_row(
    spell_1: dict[str, Any],
    spell_2: dict[str, Any],
    overlap_start: pd.Timestamp | None,
    overlap_end: pd.Timestamp | None,
    gap_start: pd.Timestamp | None,
    gap_end: pd.Timestamp | None,
) -> dict[str, Any]:
    """Format one same-club player-pair evidence row."""
    overlap_days = (
        int((overlap_end - overlap_start).days)
        if overlap_start is not None and overlap_end is not None
        else pd.NA
    )
    gap_days = (
        int((gap_end - gap_start).days)
        if gap_start is not None and gap_end is not None
        else pd.NA
    )

    return {
        "club_id": spell_1.get("club_id"),
        "club_name": spell_1.get("club_name") or spell_2.get("club_name"),
        "player_1_id": spell_1.get("player_id"),
        "player_1_name": spell_1.get("player_name"),
        "player_1_join_date": spell_1["join_date"].strftime("%Y-%m-%d"),
        "player_1_leave_date": spell_1["leave_date"].strftime("%Y-%m-%d"),
        "player_2_id": spell_2.get("player_id"),
        "player_2_name": spell_2.get("player_name"),
        "player_2_join_date": spell_2["join_date"].strftime("%Y-%m-%d"),
        "player_2_leave_date": spell_2["leave_date"].strftime("%Y-%m-%d"),
        "overlap_start_date": overlap_start.strftime("%Y-%m-%d")
        if overlap_start is not None
        else pd.NA,
        "overlap_end_date": overlap_end.strftime("%Y-%m-%d")
        if overlap_end is not None
        else pd.NA,
        "overlap_days": overlap_days,
        "gap_start_date": gap_start.strftime("%Y-%m-%d") if gap_start is not None else pd.NA,
        "gap_end_date": gap_end.strftime("%Y-%m-%d") if gap_end is not None else pd.NA,
        "gap_days": gap_days,
    }


def build_same_club_candidates(
    transfers: pd.DataFrame,
    max_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    teammate_columns = CANDIDATE_SCHEMAS["candidate_same_club_questions.csv"]
    non_overlap_columns = CANDIDATE_SCHEMAS[
        "candidate_same_club_non_overlap_questions.csv"
    ]
    spells = build_complete_club_spells(transfers)
    if spells.empty:
        return (
            pd.DataFrame(columns=teammate_columns),
            pd.DataFrame(columns=non_overlap_columns),
            0,
        )

    teammate_rows: list[dict[str, Any]] = []
    non_overlap_rows: list[dict[str, Any]] = []
    seen_teammates: set[tuple[Any, ...]] = set()
    seen_non_overlaps: set[tuple[Any, ...]] = set()

    for club_id, club_group in spells.groupby("club_id", dropna=False):
        if len(club_group) < 2:
            continue

        club_group = club_group.sort_values(
            ["player_name", "join_date", "leave_date"], na_position="last"
        )
        for spell_1, spell_2 in combinations(club_group.to_dict("records"), 2):
            if normalize_key_value(spell_1.get("player_id")) == normalize_key_value(
                spell_2.get("player_id")
            ):
                continue

            pair_key = tuple(
                sorted(
                    [
                        normalize_key_value(spell_1.get("player_id")),
                        normalize_key_value(spell_2.get("player_id")),
                    ]
                )
            )
            evidence_key = (
                club_id,
                pair_key,
                spell_1["join_date"],
                spell_1["leave_date"],
                spell_2["join_date"],
                spell_2["leave_date"],
            )

            overlap_start = max(spell_1["join_date"], spell_2["join_date"])
            overlap_end = min(spell_1["leave_date"], spell_2["leave_date"])
            has_overlap = overlap_start < overlap_end

            if has_overlap:
                if len(teammate_rows) >= max_pairs or evidence_key in seen_teammates:
                    continue
                seen_teammates.add(evidence_key)
                teammate_rows.append(
                    format_pair_row(
                        spell_1,
                        spell_2,
                        overlap_start,
                        overlap_end,
                        None,
                        None,
                    )
                )
            else:
                if len(non_overlap_rows) >= max_pairs or evidence_key in seen_non_overlaps:
                    continue
                seen_non_overlaps.add(evidence_key)
                if spell_1["leave_date"] <= spell_2["join_date"]:
                    gap_start = spell_1["leave_date"]
                    gap_end = spell_2["join_date"]
                else:
                    gap_start = spell_2["leave_date"]
                    gap_end = spell_1["join_date"]
                non_overlap_rows.append(
                    format_pair_row(
                        spell_1,
                        spell_2,
                        None,
                        None,
                        gap_start,
                        gap_end,
                    )
                )

            if len(teammate_rows) >= max_pairs and len(non_overlap_rows) >= max_pairs:
                break
        if len(teammate_rows) >= max_pairs and len(non_overlap_rows) >= max_pairs:
            break

    return (
        pd.DataFrame(teammate_rows, columns=teammate_columns),
        pd.DataFrame(non_overlap_rows, columns=non_overlap_columns),
        len(spells),
    )

def join_success_report(
    left: pd.DataFrame,
    left_column: str,
    right: pd.DataFrame,
    right_column: str,
    join_name: str,
    report: Report,
) -> None:
    left_keys = normalize_key_series(left[left_column])
    right_keys = set(normalize_key_series(right[right_column]).dropna())
    matched = left_keys.isin(right_keys)
    matched_count = int(matched.sum())
    unmatched_count = int((~matched).sum())
    success_rate = matched_count / len(left) if len(left) else 0
    unmatched_examples = sorted(left_keys.loc[~matched & left_keys.notna()].unique())[:20]

    report.add(f"{join_name}")
    report.add(f"  matched rows: {matched_count:,}")
    report.add(f"  unmatched rows: {unmatched_count:,}")
    report.add(f"  success rate: {success_rate:.2%}")
    report.add(f"  examples of unmatched IDs: {unmatched_examples}")


def analyze_join_feasibility(
    players: pd.DataFrame | None,
    clubs: pd.DataFrame | None,
    countries: pd.DataFrame | None,
    transfers: pd.DataFrame | None,
    club_country_map: pd.DataFrame,
    report: Report,
) -> None:
    report.section("8. Multi-Table Join Feasibility")

    if has_columns(
        transfers,
        ["player_id"],
        "transfers",
        report,
        "players.player_id = transfers.player_id",
    ) and has_columns(players, ["player_id"], "players", report, "player-transfer join"):
        join_success_report(
            transfers,
            "player_id",
            players,
            "player_id",
            "Player to transfer: players.player_id = transfers.player_id",
            report,
        )

    if has_columns(
        transfers,
        ["from_club_id"],
        "transfers",
        report,
        "transfers.from_club_id = clubs.club_id",
    ) and has_columns(clubs, ["club_id"], "clubs", report, "transfer from-club join"):
        join_success_report(
            transfers,
            "from_club_id",
            clubs,
            "club_id",
            "Transfer to from-club: transfers.from_club_id = clubs.club_id",
            report,
        )

    if has_columns(
        transfers,
        ["to_club_id"],
        "transfers",
        report,
        "transfers.to_club_id = clubs.club_id",
    ) and has_columns(clubs, ["club_id"], "clubs", report, "transfer to-club join"):
        join_success_report(
            transfers,
            "to_club_id",
            clubs,
            "club_id",
            "Transfer to to-club: transfers.to_club_id = clubs.club_id",
            report,
        )

    report.add("Club to country:")
    if not club_country_map.empty and valid_mask(club_country_map["country_name"]).any():
        mapped = int(valid_mask(club_country_map["country_name"]).sum())
        report.add(f"  clubs mapped to country: {mapped:,}")
        if countries is not None and "country_name" in countries.columns:
            country_keys = set(countries["country_name"].map(normalize_name))
            club_keys = club_country_map["country_name"].map(normalize_name)
            matched = club_keys.isin(country_keys)
            report.add(f"  mapped clubs matching countries.csv: {int(matched.sum()):,}")
            report.add(f"  mapped clubs not matching countries.csv: {int((~matched).sum()):,}")
    else:
        report.add(
            "  not feasible from current processed tables. clubs.csv has "
            "domestic_competition_id, but no country column and no "
            "competition-to-country table in dataset_clean/."
        )


def build_citizenship_club_country_candidates(
    players: pd.DataFrame | None,
    transfers: pd.DataFrame | None,
    club_country_map: pd.DataFrame,
) -> pd.DataFrame:
    output_columns = CANDIDATE_SCHEMAS[
        "candidate_citizenship_club_country_questions.csv"
    ]
    if (
        players is None
        or transfers is None
        or club_country_map.empty
        or not {"player_id", "name", "country_of_citizenship"}.issubset(players.columns)
    ):
        return pd.DataFrame(columns=output_columns)

    if "_transfer_date_dt" not in transfers.columns:
        return pd.DataFrame(columns=output_columns)

    window = date_window(transfers, PRIMARY_START, PRIMARY_END)
    associations = []
    if {"to_club_id", "to_club_name", "player_id"}.issubset(window.columns):
        to_assoc = window[["player_id", "to_club_id", "to_club_name", "_transfer_date_dt"]].copy()
        to_assoc = to_assoc.rename(columns={"to_club_id": "club_id", "to_club_name": "club_name"})
        associations.append(to_assoc)
    if {"from_club_id", "from_club_name", "player_id"}.issubset(window.columns):
        from_assoc = window[["player_id", "from_club_id", "from_club_name", "_transfer_date_dt"]].copy()
        from_assoc = from_assoc.rename(columns={"from_club_id": "club_id", "from_club_name": "club_name"})
        associations.append(from_assoc)
    if not associations:
        return pd.DataFrame(columns=output_columns)

    assoc = pd.concat(associations, ignore_index=True)
    assoc["_player_key"] = normalize_key_series(assoc["player_id"])
    assoc["_club_key"] = normalize_key_series(assoc["club_id"])

    player_lookup = players[["player_id", "name", "country_of_citizenship"]].copy()
    player_lookup["_player_key"] = normalize_key_series(player_lookup["player_id"])

    club_lookup = club_country_map[[
        "club_id",
        "country_name",
        "country_source",
    ]].copy()
    club_lookup["_club_key"] = normalize_key_series(club_lookup["club_id"])

    joined = assoc.merge(
        player_lookup[["_player_key", "name", "country_of_citizenship"]],
        on="_player_key",
        how="left",
    ).merge(
        club_lookup[["_club_key", "country_name", "country_source"]],
        on="_club_key",
        how="left",
    )
    joined = joined.loc[
        valid_mask(joined["country_of_citizenship"])
        & valid_mask(joined["country_name"])
    ].copy()
    if joined.empty:
        return pd.DataFrame(columns=output_columns)

    joined["evidence_transfer_date"] = joined["_transfer_date_dt"].dt.strftime("%Y-%m-%d")
    joined = joined.rename(
        columns={
            "name": "player_name",
            "country_of_citizenship": "citizenship_country",
            "country_name": "club_country",
            "country_source": "club_country_source",
        }
    )
    return joined[output_columns].drop_duplicates().head(20)


def save_candidates(
    candidates: dict[str, pd.DataFrame],
    metadata_dir: Path,
    report: Report,
) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report.section("10. Output Files")
    for filename, columns in CANDIDATE_SCHEMAS.items():
        frame = ensure_columns(candidates.get(filename), columns)
        output_path = metadata_dir / filename
        frame.to_csv(output_path, index=False)
        report.add(f"Saved {len(frame):,} rows to {output_path}")
        print(f"Saved {len(frame):,} rows to {output_path}")


def add_candidate_summary(
    candidates: dict[str, pd.DataFrame],
    report: Report,
) -> None:
    report.section("9. Candidate Examples for Each Planned Question Type")
    for filename, columns in CANDIDATE_SCHEMAS.items():
        title = filename.replace("candidate_", "").replace("_questions.csv", "")
        frame = ensure_columns(candidates.get(filename), columns)
        report.add_frame(f"{title} candidates:", frame, max_rows=20)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    report_path = args.report_path or input_dir / "metadata_report.txt"
    metadata_dir = args.metadata_dir or input_dir / "metadata"

    report = Report()
    report.section("Transfermarkt Benchmark Metadata Analysis")
    report.add(f"Input directory: {input_dir}")
    report.add(f"Report path: {report_path}")
    report.add(f"Metadata candidate directory: {metadata_dir}")

    tables = load_tables(input_dir, report)
    players = tables.get("players")
    clubs = tables.get("clubs")
    countries = tables.get("countries")
    transfers = tables.get("transfers")
    competitions = tables.get("competitions")

    all_candidates: dict[str, pd.DataFrame] = {}

    analyze_basic_table_stats(tables, report)
    all_candidates.update(analyze_players(players, report))
    all_candidates.update(analyze_countries(countries, players, clubs, report))
    club_candidates, club_country_map = analyze_clubs(
        clubs, countries, competitions, report
    )
    all_candidates.update(club_candidates)
    _, transfers_work = analyze_transfers(transfers, report)
    all_candidates.update(analyze_continuous_transfers(transfers_work, report))
    all_candidates.update(
        analyze_club_activity(transfers_work, report, args.max_same_club_pairs)
    )
    analyze_join_feasibility(
        players, clubs, countries, transfers_work, club_country_map, report
    )

    all_candidates[
        "candidate_citizenship_club_country_questions.csv"
    ] = build_citizenship_club_country_candidates(
        players, transfers_work, club_country_map
    )

    # If club-country mapping was impossible, explicitly keep the dependent
    # candidate files empty with the expected headers.
    all_candidates.setdefault(
        "candidate_club_country_questions.csv",
        pd.DataFrame(columns=CANDIDATE_SCHEMAS["candidate_club_country_questions.csv"]),
    )
    all_candidates.setdefault(
        "candidate_club_confederation_questions.csv",
        pd.DataFrame(columns=CANDIDATE_SCHEMAS["candidate_club_confederation_questions.csv"]),
    )

    add_candidate_summary(all_candidates, report)
    save_candidates(all_candidates, metadata_dir, report)
    report.write(report_path)
    print(f"Saved metadata report to {report_path}")


if __name__ == "__main__":
    main()
