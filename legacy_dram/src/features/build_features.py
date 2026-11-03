"""Master feature engineering pipeline.

Coordinates:
1. FinBERT sentiment scoring of news headlines
2. Keyword event tagging (shortage, tariff, demand_surge, supply_cut, disaster)
3. Daily news aggregation and rolling indicators
4. Stock price lags (1, 7, 14, 30 days) and rolling returns
5. RAM benchmark target price lags and rolling metrics
6. Unified dataset merging and export to data/processed/final_dataset.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Optional
import pandas as pd

from src.data.news import load_news
from src.data.prices import load_ram_prices
from src.data.stocks import load_stocks
from src.features.aggregator import aggregate_daily_news
from src.features.events import tag_events_dataframe
from src.features.sentiment import score_news_dataframe
from src.features.stock_features import build_ram_price_features, build_stock_features

logger = logging.getLogger(__name__)

PROCESSED_NEWS_PATH: Final[str] = "data/processed/news_features.csv"
FINAL_DATASET_PATH: Final[str] = "data/processed/final_dataset.csv"


def process_news_features(
    raw_news_path: str = "data/raw/news.csv",
    output_path: str = PROCESSED_NEWS_PATH,
) -> pd.DataFrame:
    """Score headlines with FinBERT, tag event types, and save processed headlines.

    Args:
        raw_news_path: Path to raw news CSV.
        output_path: Destination for scored and tagged news CSV.

    Returns:
        Processed news DataFrame.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    news_df = load_news(raw_news_path)
    logger.info("Scoring %d news headlines with FinBERT...", len(news_df))
    scored_df = score_news_dataframe(news_df, text_col="headline")

    logger.info("Tagging event types (shortage, tariff, demand_surge, supply_cut, disaster)...")
    tagged_df = tag_events_dataframe(scored_df, text_col="headline")

    tagged_df.to_csv(out_file, index=False)
    logger.info("Saved processed news headlines to %s", out_file)
    return tagged_df


def build_final_dataset(
    stocks_path: str = "data/raw/stocks.csv",
    news_path: str = "data/raw/news.csv",
    ram_prices_path: str = "data/raw/ram_prices.csv",
    output_path: str = FINAL_DATASET_PATH,
) -> pd.DataFrame:
    """Merge RAM prices, stock features, and news aggregates into a unified dataset.

    Args:
        stocks_path: Path to stocks CSV.
        news_path: Path to news CSV.
        ram_prices_path: Path to RAM prices CSV.
        output_path: Path to final merged CSV.

    Returns:
        Clean, leak-free DataFrame with all engineered features aligned by date.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load RAM target prices & compute RAM lag features
    ram_df = load_ram_prices(ram_prices_path)
    ram_feat_df = build_ram_price_features(ram_df)

    start_date = ram_feat_df["date"].min()
    end_date = ram_feat_df["date"].max()
    full_calendar_dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # 2. Load stock prices & engineer lags & rolling returns
    stocks_df = load_stocks(stocks_path)
    stock_feat_df = build_stock_features(stocks_df, full_calendar_dates=full_calendar_dates)

    # 3. Process news headlines: score sentiment and tag events
    processed_news = process_news_features(news_path, PROCESSED_NEWS_PATH)
    daily_news = aggregate_daily_news(processed_news, start_date=start_date, end_date=end_date)

    # 4. Merge all feature sources on 'date'
    merged = ram_feat_df.merge(stock_feat_df, on="date", how="inner")
    merged = merged.merge(daily_news, on="date", how="left")

    # Fill any remaining NaNs in news columns (for days without news, neutral/zero)
    for col in daily_news.columns:
        if col != "date" and col in merged.columns:
            if "sentiment" in col:
                merged[col] = merged[col].fillna(0.0)
            else:
                merged[col] = merged[col].fillna(0)

    # Drop early initial rows where 30-day lookback lag is not yet available
    # (keeps strict consistency across all models)
    merged = merged.dropna().sort_values("date").reset_index(drop=True)

    merged.to_csv(out_file, index=False)
    logger.info(
        "Final engineered dataset ready at %s: %d rows x %d columns (from %s to %s)",
        out_file,
        len(merged),
        merged.shape[1],
        merged["date"].iloc[0],
        merged["date"].iloc[-1],
    )
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset = build_final_dataset()
    print("Engineered dataset preview:")
    print(dataset.head())
    print("\nColumns:", list(dataset.columns))
