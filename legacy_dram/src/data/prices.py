"""Module for loading and managing DRAM/RAM benchmark target prices.

Loads historical RAM price data from data/raw/ram_prices.csv.
If the file is missing, generates a realistic sample dataset with clear
instructions for replacing it with user-provided market data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RAM_PRICES_PATH: Final[str] = "data/raw/ram_prices.csv"


def create_sample_ram_prices(
    file_path: str | Path = DEFAULT_RAM_PRICES_PATH,
    days: int = 1095,  # ~3 years
    end_date: str | None = None,
) -> pd.DataFrame:
    """Create a sample RAM price CSV with clear TODO comments if missing.

    Generates realistic DRAM spot/benchmark market prices (e.g., DDR4/DDR5
    8Gb chip / module equivalents in USD) reflecting cyclical memory dynamics:
    - 2023: Inventory digestion and cyclical trough (~$2.20 - $2.80)
    - 2024: Production cuts and recovery (~$2.90 - $3.80)
    - 2025-2026: AI / HBM capacity shift and memory price surges (~$3.90 - $4.90)

    Args:
        file_path: Destination path for the CSV.
        days: Number of historical days to generate.
        end_date: Last date string (YYYY-MM-DD). Defaults to today.

    Returns:
        DataFrame containing 'date' and 'price' columns.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if end_date is None:
        last_dt = pd.Timestamp.now().normalize()
    else:
        last_dt = pd.Timestamp(end_date)

    date_range = pd.date_range(end=last_dt, periods=days, freq="D")

    # Generate realistic memory cycle price curve:
    # Macro cyclical wave + medium-term trend + daily realistic noise
    t = np.linspace(0, 3 * np.pi, days)
    # Long cycle: bottom around 25-35% into the window, followed by upward supercycle
    cycle = 3.20 - 0.70 * np.cos(t * 0.7) + 0.50 * np.sin(t * 0.3)
    np.random.seed(42)
    # Daily autocorrelated random walk noise (low volatility on daily DRAM spot market)
    daily_returns = np.random.normal(loc=0.0003, scale=0.009, size=days)
    walk = np.cumprod(1 + daily_returns)
    walk = walk / walk[0]

    raw_prices = cycle * walk
    # Smooth slightly to mimic DRAM spot/contract pricing behavior
    series = pd.Series(raw_prices).rolling(window=3, min_periods=1).mean()
    prices = series.round(4).to_numpy()

    df = pd.DataFrame({
        "date": date_range.strftime("%Y-%m-%d"),
        "price": prices,
    })

    # Prepare file content with helpful TODO comments
    header_comment = (
        "# =========================================================================\n"
        "# TODO: REPLACE THIS SAMPLE WITH YOUR REAL DRAM/RAM SPOT OR CONTRACT PRICES\n"
        "# =========================================================================\n"
        "# Expected Columns:\n"
        "#   - date  : Date in YYYY-MM-DD format\n"
        "#   - price : Price in USD (e.g., DDR4 8Gb spot, DDR5 contract, or RAM kit index)\n"
        "#\n"
        "# Recommended data sources:\n"
        "#   - DRAMeXchange / TrendForce (DRAM spot and contract pricing tables)\n"
        "#   - InSpectrum memory indices\n"
        "#   - PCPartPicker / CamelCamelCamel RAM price history\n"
        "#   - Bloomberg / LSEG memory commodity trackers\n"
        "# =========================================================================\n"
    )

    csv_data = df.to_csv(index=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header_comment + csv_data)

    logger.info(
        "Created sample RAM prices CSV at %s (%d rows with clear TODO instructions)",
        path,
        len(df),
    )
    return df


def load_ram_prices(
    file_path: str | Path = DEFAULT_RAM_PRICES_PATH,
    auto_generate_sample: bool = True,
) -> pd.DataFrame:
    """Load and validate RAM target prices from CSV.

    Args:
        file_path: Path to the target price CSV.
        auto_generate_sample: If True, creates sample file if not found.

    Returns:
        DataFrame sorted by date ascending with ['date', 'price'].

    Raises:
        FileNotFoundError: If file is missing and auto_generate_sample is False.
        ValueError: If required columns ('date', 'price') are missing.
    """
    path = Path(file_path)

    if not path.exists():
        if auto_generate_sample:
            logger.warning(
                "Target price file not found at %s. Creating sample file with TODOs.",
                path,
            )
            return create_sample_ram_prices(path)
        raise FileNotFoundError(f"RAM prices file not found at: {path}")

    # Read CSV, ignoring comment lines starting with '#'
    df = pd.read_csv(path, comment="#")

    # Normalize column names (strip whitespace and lower case)
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" not in df.columns or "price" not in df.columns:
        raise ValueError(
            f"RAM prices CSV at {path} must contain 'date' and 'price' columns. "
            f"Found columns: {list(df.columns)}"
        )

    # Clean and parse
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Drop NaNs and sort
    df = df.dropna(subset=["date", "price"]).sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    logger.info(
        "Loaded %d RAM price records from %s (from %s to %s)",
        len(df),
        path,
        df["date"].iloc[0] if not df.empty else "N/A",
        df["date"].iloc[-1] if not df.empty else "N/A",
    )
    return df[["date", "price"]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ram_df = load_ram_prices()
    print("Sample RAM prices loaded successfully:")
    print(ram_df.head())
    print("Total rows:", len(ram_df))
