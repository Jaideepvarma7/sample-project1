"""Load the raw smartphone dataset.

The data comes from a public Kaggle dataset (scraped from smartprix.com):
    https://www.kaggle.com/datasets/shrutiambekar/smartphone-specifications-and-prices-in-india
    File on Kaggle: "smartphones - smartphones.csv"  (1020 phones, prices in INR)

Download it manually and save it as:
    data/raw/smartphones.csv

This module only reads that file. It never creates or fakes data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "smartphones.csv"

KAGGLE_URL = (
    "https://www.kaggle.com/datasets/shrutiambekar/"
    "smartphone-specifications-and-prices-in-india"
)
KAGGLE_FILE = "smartphones - smartphones.csv"

# Columns the Kaggle file must contain (all stored as text).
# "os" is optional: Kaggle's "selected columns" export sometimes drops it,
# and the cleaning step can derive the OS from the brand and other columns.
EXPECTED_COLUMNS = [
    "model", "price", "rating", "sim", "processor", "ram",
    "battery", "display", "camera", "card",
]
OPTIONAL_COLUMNS = ["os"]


class DatasetNotFoundError(FileNotFoundError):
    """Raised when data/raw/smartphones.csv is missing."""


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Read the raw smartphone CSV and return it as a DataFrame.

    Raises a clear error if the file is missing or does not look like the
    expected Kaggle file. Nothing is generated or filled in automatically.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(
            f"\n\nDataset not found: {path}\n\n"
            "Please download it from Kaggle and place it at that path:\n"
            f"  1. Open {KAGGLE_URL}\n"
            f"  2. Download the file \"{KAGGLE_FILE}\"\n"
            f"  3. Save it as {path}\n"
        )

    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Make sure you downloaded \"{KAGGLE_FILE}\" from {KAGGLE_URL}."
        )
    if df.empty:
        raise ValueError(f"{path} contains no rows.")

    return df


def describe(df: pd.DataFrame) -> str:
    """Short human-readable summary of the raw data."""
    lines = [
        f"Rows: {len(df)}   Columns: {len(df.columns)}",
        f"Columns: {', '.join(df.columns)}",
        f"Missing values per column: {df.isna().sum().to_dict()}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    data = load_raw()
    print(describe(data))
    print(data.head())
