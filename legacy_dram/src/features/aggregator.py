"""Module for aggregating news sentiment and event counts by day.

Computes daily metrics:
- avg sentiment score
- headline count
- event counts: count_shortage, count_tariff, count_demand_surge, count_supply_cut, count_disaster
- rolling 7-day and 14-day sentiment and event indicators
"""

from __future__ import annotations

import logging
from typing import Final, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EVENT_COLS: Final[List[str]] = [
    "event_shortage",
    "event_tariff",
    "event_demand_surge",
    "event_supply_cut",
    "event_disaster",
]


def aggregate_daily_news(
    news_df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Aggregate headline sentiment and event counts into daily metrics.

    Args:
        news_df: DataFrame with date, sentiment_score, and event_* columns.
        start_date: Optional lower date bound.
        end_date: Optional upper date bound.

    Returns:
        DataFrame with one row per calendar day and aggregated features.
    """
    df = news_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Group by date
    grouped = df.groupby("date")

    agg_dict = {
        "headline": "count",
        "sentiment_score": "mean",
    }
    for ec in EVENT_COLS:
        if ec in df.columns:
            agg_dict[ec] = "sum"

    daily = grouped.agg(agg_dict).reset_index()
    daily = daily.rename(columns={
        "headline": "news_headline_count",
        "sentiment_score": "news_avg_sentiment",
        "event_shortage": "count_shortage",
        "event_tariff": "count_tariff",
        "event_demand_surge": "count_demand_surge",
        "event_supply_cut": "count_supply_cut",
        "event_disaster": "count_disaster",
    })

    # Positive and negative ratio calculation
    if "sentiment_label" in df.columns:
        pos_counts = df[df["sentiment_label"] == "positive"].groupby("date").size()
        neg_counts = df[df["sentiment_label"] == "negative"].groupby("date").size()
        daily["count_pos_sentiment"] = daily["date"].map(pos_counts).fillna(0).astype(int)
        daily["count_neg_sentiment"] = daily["date"].map(neg_counts).fillna(0).astype(int)
        daily["news_pos_ratio"] = (
            daily["count_pos_sentiment"] / daily["news_headline_count"]
        ).round(4)
        daily["news_neg_ratio"] = (
            daily["count_neg_sentiment"] / daily["news_headline_count"]
        ).round(4)

    # Reindex over continuous date range so days with no news have 0 count and neutral sentiment
    if not daily.empty:
        min_date = start_date or daily["date"].min()
        max_date = end_date or daily["date"].max()
        full_dates = pd.date_range(start=min_date, end=max_date, freq="D").strftime("%Y-%m-%d")
        daily = daily.set_index("date").reindex(full_dates).reset_index()
        daily = daily.rename(columns={"index": "date"})

        # Fill missing news counts with 0, sentiment with 0 (neutral)
        daily["news_headline_count"] = daily["news_headline_count"].fillna(0).astype(int)
        daily["news_avg_sentiment"] = daily["news_avg_sentiment"].fillna(0.0)

        for col in [
            "count_shortage",
            "count_tariff",
            "count_demand_surge",
            "count_supply_cut",
            "count_disaster",
            "count_pos_sentiment",
            "count_neg_sentiment",
            "news_pos_ratio",
            "news_neg_ratio",
        ]:
            if col in daily.columns:
                daily[col] = daily[col].fillna(0)

        # Calculate rolling features
        daily["news_sentiment_roll_7d"] = (
            daily["news_avg_sentiment"].rolling(window=7, min_periods=1).mean().round(4)
        )
        daily["news_sentiment_roll_14d"] = (
            daily["news_avg_sentiment"].rolling(window=14, min_periods=1).mean().round(4)
        )
        daily["count_shortage_roll_7d"] = (
            daily["count_shortage"].rolling(window=7, min_periods=1).sum()
        )
        daily["count_demand_surge_roll_7d"] = (
            daily["count_demand_surge"].rolling(window=7, min_periods=1).sum()
        )
        daily["count_supply_cut_roll_7d"] = (
            daily["count_supply_cut"].rolling(window=7, min_periods=1).sum()
        )

    logger.info("Aggregated daily news features across %d days.", len(daily))
    return daily
