"""Feature engineering package for gadget-price-predictor."""

from src.features.aggregator import aggregate_daily_news
from src.features.build_features import build_final_dataset, process_news_features
from src.features.events import get_event_tags_list, tag_event_types, tag_events_dataframe
from src.features.sentiment import FinBERTSentimentScorer, score_news_dataframe
from src.features.stock_features import build_ram_price_features, build_stock_features

__all__ = [
    "aggregate_daily_news",
    "build_final_dataset",
    "process_news_features",
    "get_event_tags_list",
    "tag_event_types",
    "tag_events_dataframe",
    "FinBERTSentimentScorer",
    "score_news_dataframe",
    "build_ram_price_features",
    "build_stock_features",
]
