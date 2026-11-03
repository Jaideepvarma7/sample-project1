"""Master pipeline runner for gadget-price-predictor.

Executes the complete machine learning workflow end-to-end with a single command:
    python run_pipeline.py

Pipeline steps:
1. Data Collection:
   - Stocks (MU, 005930.KS, 000660.KS, TSM, NVDA) via yfinance -> data/raw/stocks.csv
   - Global News via GDELT DOC 2.0 API -> data/raw/news.csv
   - DRAM/RAM target prices -> data/raw/ram_prices.csv
2. Feature Engineering:
   - FinBERT sentiment scoring (ProsusAI/finbert)
   - Keyword event tagging (shortage, tariff, demand_surge, supply_cut, disaster)
   - Daily news aggregation and rolling indicators
   - Stock lags (1, 7, 14, 30 days) and rolling returns
   - Target price lags and rolling returns
   - Unified dataset merge -> data/processed/final_dataset.csv
3. Model Training & Evaluation:
   - 14-day forward return labeling (UP >+3%, DOWN <-3%, FLAT)
   - Walk-forward validation (strict time order, expanding window, no shuffling)
   - Benchmark comparison against 'tomorrow = today' persistence baseline
   - Confusion matrix, Accuracy, Macro/Weighted F1
   - SHAP feature importance analysis
   - Model artifact export -> models/lgbm_model.pkl & models/metadata.json
4. Latest Forecast Generation & Dashboard readiness
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PipelineRunner")


def run_data_collection(fetch_live_gdelt: bool = True) -> None:
    """Step 1: Download or load raw stocks, news, and RAM benchmark prices."""
    logger.info("=" * 65)
    logger.info("STEP 1: DATA COLLECTION (src/data/)")
    logger.info("=" * 65)

    from src.data.stocks import fetch_stocks_data
    from src.data.news import fetch_all_news
    from src.data.prices import load_ram_prices

    logger.info("[1/3] Downloading 3-year daily stock prices (MU, 005930.KS, 000660.KS, TSM, NVDA)...")
    stocks_df = fetch_stocks_data()
    logger.info(" -> Stocks data ready: %d records.", len(stocks_df))

    logger.info("[2/3] Fetching global semiconductor news headlines from GDELT DOC 2.0 API...")
    news_df = fetch_all_news(fetch_live_gdelt=fetch_live_gdelt)
    logger.info(" -> News data ready: %d headlines.", len(news_df))

    logger.info("[3/3] Loading target RAM prices from data/raw/ram_prices.csv...")
    ram_df = load_ram_prices()
    logger.info(" -> RAM target prices ready: %d days.", len(ram_df))


def run_feature_engineering() -> None:
    """Step 2: Score headlines with FinBERT, tag events, compute lags and merge."""
    logger.info("=" * 65)
    logger.info("STEP 2: FEATURE ENGINEERING (src/features/)")
    logger.info("=" * 65)

    from src.features.build_features import build_final_dataset

    logger.info("Running FinBERT sentiment scoring, event tagging, and lag computation...")
    final_df = build_final_dataset()
    logger.info(
        " -> Final engineered dataset generated: %d samples, %d features.",
        len(final_df),
        final_df.shape[1] - 1,
    )


def run_model_training_and_eval(n_splits: int = 5) -> None:
    """Step 3: Train LightGBM with walk-forward validation and evaluate against baseline."""
    logger.info("=" * 65)
    logger.info("STEP 3: MODEL TRAINING & WALK-FORWARD EVALUATION (src/model/)")
    logger.info("=" * 65)

    from src.model.train import train_and_evaluate

    train_and_evaluate(n_splits=n_splits)


def print_latest_forecast() -> None:
    """Step 4: Output the live 14-day forecast for memory prices."""
    logger.info("=" * 65)
    logger.info("STEP 4: LATEST LIVE 14-DAY FORECAST")
    logger.info("=" * 65)

    from src.model.train import predict_latest_forecast

    forecast = predict_latest_forecast()
    pred = forecast["prediction"]
    conf = forecast["confidence"]
    price = forecast["current_price"]
    as_of = forecast["as_of_date"]
    probs = forecast["probabilities"]

    icon = "📈" if pred == "UP" else ("📉" if pred == "DOWN" else "➡️")
    delta_str = "> +3.0%" if pred == "UP" else ("< -3.0%" if pred == "DOWN" else "Within ±3.0%")

    print("\n" + "*" * 65)
    print(f"  {icon} DRAM/RAM 14-DAY PRICE PREDICTION: {pred} ({delta_str})")
    print("*" * 65)
    print(f"  • As of Date        : {as_of}")
    print(f"  • Current Spot Price: ${price:.4f} USD")
    print(f"  • Model Confidence  : {conf * 100:.2f}%")
    print("  • Class Probabilities:")
    print(f"      - UP   (>+3%) : {probs['UP'] * 100:>6.2f}%")
    print(f"      - FLAT (±3%)  : {probs['FLAT'] * 100:>6.2f}%")
    print(f"      - DOWN (<-3%) : {probs['DOWN'] * 100:>6.2f}%")
    print("*" * 65 + "\n")


def main() -> None:
    """Parse CLI options and run the complete end-to-end pipeline."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end DRAM/RAM price prediction pipeline."
    )
    parser.add_argument(
        "--skip-live-gdelt",
        action="store_true",
        help="Skip querying live GDELT API to avoid network timeouts/rate-limits.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of expanding-window walk-forward splits (default: 5).",
    )
    args = parser.parse_args()

    start_time = time.time()
    logger.info("Starting gadget-price-predictor end-to-end pipeline...")

    try:
        # Step 1: Collect raw data
        run_data_collection(fetch_live_gdelt=not args.skip_live_gdelt)

        # Step 2: Engineer features
        run_feature_engineering()

        # Step 3: Train & Walk-Forward Evaluate
        run_model_training_and_eval(n_splits=args.splits)

        # Step 4: Display Latest Forecast
        print_latest_forecast()

        elapsed = time.time() - start_time
        logger.info("Pipeline completed successfully in %.1f seconds!", elapsed)
        print("💡 To view the interactive dashboard, run:")
        print("    streamlit run app.py\n")

    except Exception as exc:
        logger.exception("Pipeline failed with error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
