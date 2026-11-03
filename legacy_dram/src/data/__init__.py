"""Data collection and loading package for gadget-price-predictor."""

from src.data.news import fetch_all_news, load_news
from src.data.prices import create_sample_ram_prices, load_ram_prices
from src.data.stocks import fetch_stocks_data, load_stocks

__all__ = [
    "fetch_all_news",
    "load_news",
    "create_sample_ram_prices",
    "load_ram_prices",
    "fetch_stocks_data",
    "load_stocks",
]
