"""Run benchmark modification scripts from the project root.

Examples:
    python run_modification.py
    python run_modification.py 1
    python run_modification.py level_2
    python run_modification.py 1 3
    python run_modification.py --select
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

LEVEL_SCRIPTS = {
    "level_1": PROJECT_ROOT / "modification" / "level_1.py",
    "level_2": PROJECT_ROOT / "modification" / "level_2.py",
    "level_3": PROJECT_ROOT / "modification" / "level_3.py",
    "level_4": PROJECT_ROOT / "modification" / "level_4.py",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Run all or selected scripts in modification/."
    )
    parser.add_argument(
        "levels",
        nargs="*",
        help=(
            "Levels to run. Use all, level_1, level_2, level_3, level_4, "
            "or shorthand numbers like 1 3. Defaults to all."
        ),
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Prompt interactively for which level(s) to run.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available level scripts without running them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later levels if one level fails.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Defaults to the current interpreter.",
    )
    return parser.parse_args()


def list_levels() -> None:
    """Print available level scripts."""
    print("Available modification scripts:")
    for level, path in LEVEL_SCRIPTS.items():
        print(f"  {level}: {path.relative_to(PROJECT_ROOT)}")


def split_level_tokens(tokens: list[str]) -> list[str]:
    """Allow comma-separated or space-separated level selections."""
    split_tokens: list[str] = []
    for token in tokens:
        split_tokens.extend(part.strip() for part in token.split(",") if part.strip())
    return split_tokens


def normalize_level(token: str) -> str:
    """Convert input like 1 or level-1 into the canonical level name."""
    normalized = token.strip().lower().replace("-", "_")
    if normalized.startswith("level_"):
        return normalized
    if normalized.isdigit():
        return f"level_{normalized}"
    return normalized


def resolve_levels(tokens: list[str]) -> list[str]:
    """Resolve user tokens into ordered level names."""
    tokens = split_level_tokens(tokens)
    if not tokens or any(token.lower() == "all" for token in tokens):
        return list(LEVEL_SCRIPTS)

    selected: list[str] = []
    invalid: list[str] = []
    for token in tokens:
        level = normalize_level(token)
        if level not in LEVEL_SCRIPTS:
            invalid.append(token)
            continue
        if level not in selected:
            selected.append(level)

    if invalid:
        valid = ", ".join(["all", *LEVEL_SCRIPTS.keys(), "1", "2", "3", "4"])
        raise ValueError(f"Unknown level selection {invalid}. Valid selections: {valid}")

    return selected


def prompt_for_levels() -> list[str]:
    """Prompt the user for levels to run."""
    list_levels()
    raw = input("Run which level(s)? Enter all, one level, or comma-separated values: ")
    return resolve_levels([raw])


def run_level(level: str, python_executable: str, dry_run: bool) -> int:
    """Run one level script and return its exit code."""
    script_path = LEVEL_SCRIPTS[level]
    if not script_path.exists():
        raise FileNotFoundError(f"Modification script not found: {script_path}")

    command = [python_executable, "-B", str(script_path.relative_to(PROJECT_ROOT))]
    print()
    print(f"Running {level}: {' '.join(command)}")
    if dry_run:
        return 0

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def main() -> int:
    """Run selected modification scripts."""
    args = parse_args()

    if args.list:
        list_levels()
        return 0

    try:
        selected_levels = prompt_for_levels() if args.select else resolve_levels(args.levels)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print("Selected levels: " + ", ".join(selected_levels))

    failures: list[tuple[str, int]] = []
    for level in selected_levels:
        try:
            return_code = run_level(level, args.python, args.dry_run)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        if return_code == 0:
            continue

        failures.append((level, return_code))
        print(f"{level} failed with exit code {return_code}", file=sys.stderr)
        if not args.continue_on_error:
            break

    if failures:
        print()
        print("Modification run failed:", file=sys.stderr)
        for level, return_code in failures:
            print(f"  {level}: exit code {return_code}", file=sys.stderr)
        return failures[0][1]

    print()
    print("Modification run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
