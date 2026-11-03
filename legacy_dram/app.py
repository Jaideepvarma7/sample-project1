"""Streamlit Web Dashboard for Gadget RAM Price Predictor.

Visualizes:
1. Latest 14-Day Forecast (UP, DOWN, FLAT) with Confidence Score and Class Probabilities
2. Historical RAM Benchmark Spot Price Chart with moving averages & forecast cone
3. Top 10 News Headlines that influenced the prediction (FinBERT sentiment & event tags)
4. Semiconductor Stock Influences (MU, 005930.KS, 000660.KS, TSM, NVDA)
5. Top SHAP / Model Feature Importances
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

# Paths to artifacts
MODEL_METADATA_PATH = Path("models/metadata.json")
FINAL_DATASET_PATH = Path("data/processed/final_dataset.csv")
PROCESSED_NEWS_PATH = Path("data/processed/news_features.csv")
RAW_RAM_PRICES_PATH = Path("data/raw/ram_prices.csv")
RAW_STOCKS_PATH = Path("data/raw/stocks.csv")


def load_dashboard_data() -> Dict[str, Any]:
    """Load metadata, datasets, and recent news for dashboard visualization."""
    data: Dict[str, Any] = {}

    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            data["metadata"] = json.load(f)
    else:
        data["metadata"] = None

    if FINAL_DATASET_PATH.exists():
        data["dataset"] = pd.read_csv(FINAL_DATASET_PATH)
    else:
        data["dataset"] = pd.DataFrame()

    if PROCESSED_NEWS_PATH.exists():
        data["news"] = pd.read_csv(PROCESSED_NEWS_PATH)
    else:
        data["news"] = pd.DataFrame()

    if RAW_STOCKS_PATH.exists():
        data["stocks"] = pd.read_csv(RAW_STOCKS_PATH)
    else:
        data["stocks"] = pd.DataFrame()

    return data


def run_streamlit_app() -> None:
    """Render the full Streamlit dashboard."""
    import streamlit as st  # type: ignore

    st.set_page_config(
        page_title="DRAM / RAM Price Predictor",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    data = load_dashboard_data()
    dataset = data["dataset"]
    metadata = data["metadata"]
    news_df = data["news"]
    stocks_df = data["stocks"]

    # Sidebar
    with st.sidebar:
        st.title("⚡ Settings & Info")
        st.markdown(
            """
            **gadget-price-predictor** forecasts whether DRAM/RAM benchmark spot prices
            will move **UP (>+3%)**, **DOWN (<-3%)**, or remain **FLAT** over the next **14 days**.
            
            **Input Signals:**
            - 📈 Global Tech Stocks: MU, Samsung, SK Hynix, TSM, NVDA
            - 📰 Global News: GDELT DOC 2.0 API + FinBERT NLP
            - 🏷️ Event Tagging: Shortages, Tariffs, Demand Surges, Supply Cuts, Disasters
            - 🤖 Algorithm: LightGBM with Walk-Forward Validation
            """
        )
        st.divider()
        lookback_days = st.slider("Lookback Chart Days", min_value=30, max_value=365, value=180)

        if st.button("🔄 Refresh Data & Retrain"):
            with st.spinner("Re-running end-to-end pipeline..."):
                from run_pipeline import main as run_pipeline_main
                run_pipeline_main()
                st.success("Pipeline refreshed successfully!")
                st.rerun()

    # Header
    st.markdown(
        """
        # ⚡ DRAM / RAM 14-Day Price Predictor
        **Forecasting global memory chip spot price direction using AI, semiconductor stock dynamics & FinBERT sentiment.**
        """
    )

    if dataset.empty or metadata is None:
        st.warning(
            "⚠️ Model or dataset artifacts not found! Please run `python run_pipeline.py` first."
        )
        return

    # Extract latest prediction
    from src.model.train import predict_latest_forecast
    latest = predict_latest_forecast()

    pred = latest["prediction"]
    conf = latest["confidence"]
    probs = latest["probabilities"]
    current_price = latest["current_price"]
    as_of_date = latest["as_of_date"]

    # Color badge for prediction
    pred_color = "#10B981" if pred == "UP" else ("#EF4444" if pred == "DOWN" else "#3B82F6")
    pred_icon = "📈" if pred == "UP" else ("📉" if pred == "DOWN" else "➡️")

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Latest Spot Price",
            value=f"${current_price:.2f}",
            delta=f"As of {as_of_date}",
        )

    with col2:
        st.markdown(
            f"""
            <div style="background-color: {pred_color}15; border: 2px solid {pred_color}; border-radius: 8px; padding: 10px; text-align: center;">
                <span style="font-size: 13px; color: gray; font-weight: 600;">14-DAY FORECAST</span><br>
                <span style="font-size: 26px; font-weight: bold; color: {pred_color};">{pred_icon} {pred}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.metric(
            label="Model Confidence",
            value=f"{conf * 100:.1f}%",
            delta=f"P({pred})",
        )

    with col4:
        wf_acc = metadata.get("walk_forward_metrics", {}).get("accuracy", 0.0)
        wf_f1 = metadata.get("walk_forward_metrics", {}).get("macro_f1", 0.0)
        st.metric(
            label="Walk-Forward F1 (Macro)",
            value=f"{wf_f1:.3f}",
            delta=f"Accuracy {wf_acc*100:.1f}%",
        )

    st.markdown("---")

    # Probabilities & Price Chart Row
    col_chart, col_probs = st.columns([2, 1])

    with col_chart:
        st.subheader("📊 DRAM Benchmark Price & 14-Day Trend")
        chart_data = dataset.tail(lookback_days).copy()
        chart_data["date"] = pd.to_datetime(chart_data["date"])
        chart_data["MA_7d"] = chart_data["price"].rolling(7).mean()
        chart_data["MA_30d"] = chart_data["price"].rolling(30).mean()

        plot_df = chart_data.set_index("date")[["price", "MA_7d", "MA_30d"]]
        st.line_chart(plot_df, height=320)

        # Forecast corridor note
        delta_str = "> +3.0%" if pred == "UP" else ("< -3.0%" if pred == "DOWN" else "Within ±3.0%")
        st.caption(
            f"ℹ️ **Projected 14-Day Movement:** {pred} ({delta_str} from ${current_price:.2f})"
        )

    with col_probs:
        st.subheader("🎯 Class Probabilities")
        st.markdown(f"**UP (>+3%):** `{probs['UP']*100:.1f}%`")
        st.progress(float(probs["UP"]))
        st.markdown(f"**FLAT (±3%):** `{probs['FLAT']*100:.1f}%`")
        st.progress(float(probs["FLAT"]))
        st.markdown(f"**DOWN (<-3%):** `{probs['DOWN']*100:.1f}%`")
        st.progress(float(probs["DOWN"]))

        st.markdown("#### 🏆 Benchmark Comparison")
        base = metadata.get("baseline_metrics", {})
        flat_acc = base.get("flat_baseline", {}).get("accuracy", 0.0)
        st.write(f"- **Tomorrow = Today (Flat):** {flat_acc*100:.1f}% acc")
        st.write(f"- **LightGBM Time-Split:** {wf_acc*100:.1f}% acc")

    st.markdown("---")

    # Section 3: Top 10 Influential News Headlines
    st.subheader("📰 Top 10 News Headlines Influencing Sentiment")
    st.markdown(
        "Recent global headlines scored by **FinBERT** (`ProsusAI/finbert`) with domain event tagging:"
    )

    if not news_df.empty:
        # Sort by date descending and grab top recent influential ones
        recent_news = news_df.sort_values("date", ascending=False).head(10).copy()

        for _, item in recent_news.iterrows():
            sentiment = item.get("sentiment_label", "neutral").upper()
            score = item.get("sentiment_score", 0.0)
            date_str = item.get("date", "N/A")
            source = item.get("source", "N/A")
            title = item.get("headline", "")
            url = item.get("url", "#")

            # Collect active tags
            tags = []
            for ev in ["shortage", "tariff", "demand_surge", "supply_cut", "disaster"]:
                if item.get(f"event_{ev}", 0) == 1:
                    tags.append(f"`{ev}`")
            tag_display = " ".join(tags) if tags else "`general`"

            # Badge color
            sent_badge = (
                "🟢 POSITIVE"
                if sentiment == "POSITIVE"
                else ("🔴 NEGATIVE" if sentiment == "NEGATIVE" else "⚪ NEUTRAL")
            )

            st.markdown(
                f"- **[{title}]({url})**  \n"
                f"  📅 *{date_str}* | 🌐 *{source}* | {sent_badge} (`{score:+.2f}`) | Tags: {tag_display}"
            )
    else:
        st.info("No news headlines found in data/processed/news_features.csv.")

    st.markdown("---")

    # Section 4: Key Feature Importances & Stock Signals
    col_feat, col_stocks = st.columns([1, 1])

    with col_feat:
        st.subheader("🧠 Top Feature Importances (SHAP)")
        top_feats = metadata.get("top_features", [])
        if top_feats:
            feat_df = pd.DataFrame(top_feats).rename(
                columns={"rank": "Rank", "feature": "Feature", "importance": "Importance"}
            )
            feat_df["Importance %"] = (feat_df["Importance"] * 100).round(2).astype(str) + "%"
            st.dataframe(feat_df[["Rank", "Feature", "Importance %"]], height=320, use_container_width=True)
        else:
            st.write("Feature importance data unavailable.")

    with col_stocks:
        st.subheader("📈 Semiconductor Stock Overview")
        if not stocks_df.empty:
            # Pivot recent close prices for visualization
            s_df = stocks_df.copy()
            s_df["date"] = pd.to_datetime(s_df["date"])
            s_pivot = s_df.pivot(index="date", columns="ticker", values="close").tail(lookback_days)
            # Normalize to 100 for easy relative comparison
            normalized = (s_pivot / s_pivot.iloc[0]) * 100
            st.line_chart(normalized, height=320)
            st.caption("Normalized performance (Base = 100): Micron (MU), Samsung, SK Hynix, TSMC, Nvidia")
        else:
            st.write("Stock data unavailable.")


def main() -> None:
    """Entry point supporting both Streamlit CLI and direct python execution."""
    try:
        import streamlit as st  # type: ignore
        # If running via 'streamlit run app.py', execute Streamlit UI
        run_streamlit_app()
    except ImportError:
        # Fallback terminal execution if streamlit is not yet installed
        print("=" * 65)
        print("  ⚡ DRAM / RAM PRICE PREDICTOR - CLI SUMMARY")
        print("=" * 65)
        from src.model.train import predict_latest_forecast
        try:
            latest = predict_latest_forecast()
            print(f"Latest RAM Price: ${latest['current_price']:.4f} (as of {latest['as_of_date']})")
            print(f"14-Day Forecast : {latest['prediction']} (Confidence: {latest['confidence']*100:.1f}%)")
            print("Probabilities   :")
            for k, v in latest["probabilities"].items():
                print(f"  - {k:<5}: {v*100:.2f}%")
        except Exception as e:
            print("To launch the Streamlit dashboard, install streamlit: pip install streamlit")
            print(f"Details: {e}")
        print("\nRun: streamlit run app.py to view the full interactive dashboard.")
        print("=" * 65)


if __name__ == "__main__":
    main()
