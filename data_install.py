"""Download raw Kaggle datasets into raw_data/."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

import kagglehub


DATASETS = [
    "davidcariboo/player-scores",
    "dataanalyst001/all-capital-cities-in-the-world",
]

DEFAULT_OUTPUT_DIR = Path("raw_data")


def safe_extract_zip(archive_path: Path, output_dir: Path) -> None:
    """Extract a zip archive while preventing paths from escaping output_dir."""
    output_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if not target.is_relative_to(output_root):
                raise ValueError(f"Unsafe archive path: {member.filename}")
        archive.extractall(output_dir)


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    """Extract a tar archive while preventing paths from escaping output_dir."""
    output_root = output_dir.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (output_dir / member.name).resolve()
            if not target.is_relative_to(output_root):
                raise ValueError(f"Unsafe archive path: {member.name}")
        archive.extractall(output_dir)


def copy_directory_contents(source_dir: Path, output_dir: Path) -> None:
    """Install every file from source_dir into output_dir, preserving subfolders."""
    for source_path in source_dir.iterdir():
        if source_path.is_dir():
            destination_path = output_dir / source_path.name
            destination_path.mkdir(parents=True, exist_ok=True)
            copy_directory_contents(source_path, destination_path)
        elif zipfile.is_zipfile(source_path):
            safe_extract_zip(source_path, output_dir)
        elif tarfile.is_tarfile(source_path):
            safe_extract_tar(source_path, output_dir)
        else:
            shutil.copy2(source_path, output_dir / source_path.name)


def install_downloaded_path(downloaded_path: Path, output_dir: Path) -> None:
    """Copy or unpack a Kaggle download into output_dir."""
    if downloaded_path.is_dir():
        copy_directory_contents(downloaded_path, output_dir)
    elif zipfile.is_zipfile(downloaded_path):
        safe_extract_zip(downloaded_path, output_dir)
    elif tarfile.is_tarfile(downloaded_path):
        safe_extract_tar(downloaded_path, output_dir)
    else:
        shutil.copy2(downloaded_path, output_dir / downloaded_path.name)


def download_dataset(dataset: str, output_dir: Path) -> None:
    """Download one Kaggle dataset and install its files into output_dir."""
    print(f"\nDownloading {dataset}")
    downloaded_path = Path(kagglehub.dataset_download(dataset))
    print(f"Downloaded to Kaggle cache: {downloaded_path}")

    install_downloaded_path(downloaded_path, output_dir)
    print(f"Installed files into: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw CSV data into raw_data/.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where raw files will be installed. Defaults to raw_data/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        download_dataset(dataset, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
