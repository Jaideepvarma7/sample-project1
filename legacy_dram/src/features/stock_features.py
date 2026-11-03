"""Module for engineering stock price lag and rolling return features.

Generates for each target stock (MU, 005930.KS, 000660.KS, TSM, NVDA):
- Lag features: close price at t-1, t-7, t-14, t-30
- Rolling returns: percentage returns over 1, 7, 14, 30 days
- Rolling volatility: 14-day rolling standard deviation of daily returns
"""

from __future__ import annotations

import logging
import re
from typing import Final, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LAGS: Final[List[int]] = [1, 7, 14, 30]


def clean_ticker_name(ticker: str) -> str:
    """Normalize ticker string for clean column naming (e.g. 005930.KS -> 005930_KS)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", ticker)


def build_stock_features(
    stocks_df: pd.DataFrame,
    lags: List[int] = LAGS,
    full_calendar_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Calculate lag features and rolling returns for each stock.

    Args:
        stocks_df: DataFrame with ['date', 'ticker', 'close', ...].
        lags: Lag horizons in days (default: [1, 7, 14, 30]).
        full_calendar_dates: Optional complete date range to forward-fill weekends/holidays.

    Returns:
        Wide DataFrame with one row per date and engineered stock columns.
    """
    df = stocks_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Pivot to wide format: index=date, columns=ticker, values=close
    pivoted = df.pivot(index="date", columns="ticker", values="close")

    # If full calendar dates given, reindex and forward-fill non-trading days
    if full_calendar_dates is not None:
        pivoted = pivoted.reindex(full_calendar_dates)
    pivoted = pivoted.ffill().bfill()

    feature_dfs: List[pd.DataFrame] = []

    for ticker in pivoted.columns:
        norm_name = clean_ticker_name(ticker)
        series = pivoted[ticker]

        t_df = pd.DataFrame(index=pivoted.index)
        t_df[f"{norm_name}_close"] = series

        # Lags and Rolling Returns
        for lag in lags:
            lagged_series = series.shift(lag)
            t_df[f"{norm_name}_close_lag_{lag}"] = lagged_series
            # Rolling return: (P_t - P_{t-lag}) / P_{t-lag}
            t_df[f"{norm_name}_return_{lag}d"] = (
                (series - lagged_series) / lagged_series
            ).round(5)

        # 1-day daily return for volatility calculation
        daily_ret = series.pct_change()
        t_df[f"{norm_name}_vol_14d"] = (
            daily_ret.rolling(window=14, min_periods=5).std().round(5)
        )
        t_df[f"{norm_name}_vol_30d"] = (
            daily_ret.rolling(window=30, min_periods=10).std().round(5)
        )

        feature_dfs.append(t_df)

    combined = pd.concat(feature_dfs, axis=1)
    combined = combined.reset_index().rename(columns={"index": "date"})
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")

    logger.info(
        "Engineered %d stock features across %d dates.",
        combined.shape[1] - 1,
        len(combined),
    )
    return combined


def build_ram_price_features(
    ram_df: pd.DataFrame,
    lags: List[int] = LAGS,
) -> pd.DataFrame:
    """Calculate lag features and rolling returns for the target RAM price.

    Args:
        ram_df: DataFrame with ['date', 'price'].
        lags: Lag horizons in days (default: [1, 7, 14, 30]).

    Returns:
        DataFrame with RAM lags and rolling returns.
    """
    df = ram_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    series = df["price"]

    for lag in lags:
        lagged = series.shift(lag)
        df[f"ram_price_lag_{lag}"] = lagged
        df[f"ram_return_{lag}d"] = ((series - lagged) / lagged).round(5)

    daily_ret = series.pct_change()
    df["ram_vol_14d"] = daily_ret.rolling(window=14, min_periods=5).std().round(5)
    df["ram_vol_30d"] = daily_ret.rolling(window=30, min_periods=10).std().round(5)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df
