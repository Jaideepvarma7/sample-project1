# ⚡ Gadget Price Predictor: 14-Day DRAM / RAM Direction Forecasting

A production-ready Python project that predicts whether global **DRAM / RAM benchmark spot prices** will go **UP (>+3%)**, **DOWN (<-3%)**, or remain **FLAT** over the next **14 days**, combining semiconductor stock market dynamics with global macroeconomic news sentiment.

---

## 📌 Project Overview

DRAM and memory chip prices are notoriously cyclical and sensitive to both supply-chain shocks and hyperscaler demand. This system models price movement by synthesizing:
1. **Semiconductor Stock Signals**: Daily prices and returns from the world's leading memory producers and AI catalysts:
   - `MU`: Micron Technology
   - `005930.KS`: Samsung Electronics
   - `000660.KS`: SK Hynix
   - `TSM`: Taiwan Semiconductor Manufacturing Co. (TSMC)
   - `NVDA`: Nvidia
2. **Global News Sentiment & Events**: Real-time news headlines fetched from the **GDELT DOC 2.0 API** for keywords:
   - `"DRAM"`, `"memory chip"`, `"semiconductor shortage"`, `"chip tariff"`, `"HBM"`, `"export ban"`, `"fab fire"`
3. **FinBERT NLP (`ProsusAI/finbert`)**: Scores financial sentiment into Positive, Negative, and Neutral probabilities.
4. **Domain Event Tagging**: Automated tagging for `shortage`, `tariff`, `demand_surge`, `supply_cut`, and `disaster`.
5. **LightGBM Time-Series Classifier**: Evaluated strictly using **Walk-Forward Validation** (expanding window, no shuffling) and compared against a **"tomorrow = today"** baseline with **SHAP** feature attribution.
6. **Streamlit Interactive Dashboard**: Visualizes live 14-day predictions, class probabilities, historical price charts, and the top 10 influential headlines.

---

## 🗂️ Project Directory Structure

```text
gadget-price-predictor/
├── app.py                      # Interactive Streamlit dashboard
├── run_pipeline.py             # Single command running end-to-end pipeline
├── requirements.txt            # Project dependencies
├── README.md                   # Step-by-step documentation
├── data/
│   ├── raw/
│   │   ├── stocks.csv          # 3-year daily OHLCV for MU, Samsung, SK Hynix, TSMC, NVDA
│   │   ├── news.csv            # Headlines, dates, sources, and URLs from GDELT
│   │   └── ram_prices.csv      # Target RAM benchmark prices with TODO replacement comments
│   └── processed/
│       ├── news_features.csv   # Headlines scored with FinBERT & tagged with event rules
│       └── final_dataset.csv   # Merged lag features, rolling returns & daily aggregates
├── models/
│   ├── lgbm_model.pkl          # Trained LightGBM model artifact
│   └── metadata.json           # Model validation metrics, top SHAP features & config
└── src/
    ├── data/
    │   ├── prices.py           # RAM target prices loader & sample generator
    │   ├── stocks.py           # yfinance multi-stock downloader & cache
    │   └── news.py             # GDELT DOC 2.0 API client with retries & rate limiting
    ├── features/
    │   ├── sentiment.py        # FinBERT (ProsusAI/finbert) sentiment scorer
    │   ├── events.py           # Rule-based event tagger (shortage, tariff, etc.)
    │   ├── aggregator.py       # Daily news sentiment & event count aggregation
    │   ├── stock_features.py   # Lags (1, 7, 14, 30d), rolling returns & volatilities
    │   └── build_features.py   # Master feature engineering orchestration
    └── model/
        ├── labeler.py          # 14-day forward return labeling (UP, DOWN, FLAT)
        ├── classifier.py       # LightGBM classifier with fallback support
        ├── walk_forward.py     # Time-based cross-validation (no shuffling)
        ├── evaluate.py         # Metrics, confusion matrix, baseline & SHAP analysis
        └── train.py            # Model training & serialization
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, 3.13, or 3.14
- Recommended: Virtual environment (`venv` or `conda`)

```bash
# Clone or navigate to the project directory
cd "Gadget price"

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # On macOS / Linux
# .\venv\Scripts\activate      # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Running Everything in One Command

Run the complete pipeline end-to-end:

```bash
python run_pipeline.py
```

This single command automatically:
1. **Collects Stock Data**: Downloads 3 years of daily prices for `MU`, `005930.KS`, `000660.KS`, `TSM`, `NVDA`.
2. **Fetches Global News**: Queries the GDELT DOC 2.0 API with retry backoff for memory and semiconductor keywords.
3. **Prepares Target Prices**: Loads `data/raw/ram_prices.csv` (or generates a sample file if missing).
4. **Scores Sentiments & Events**: Runs FinBERT inference on headlines and tags `shortage`, `tariff`, `demand_surge`, `supply_cut`, and `disaster`.
5. **Engineers Lags**: Calculates 1, 7, 14, and 30-day price lags and rolling returns for each stock and memory benchmark.
6. **Trains LightGBM**: Executes expanding-window Walk-Forward Validation without shuffling.
7. **Benchmarks**: Compares LightGBM against the "tomorrow = today" baseline.
8. **Reports Metrics**: Prints out-of-fold Accuracy, Macro F1, Confusion Matrix, and top SHAP feature importances.
9. **Outputs Live Prediction**: Displays the current 14-day forecast (UP, DOWN, or FLAT) and confidence.

---

## 🖥️ Launching the Streamlit Web Dashboard

Once `run_pipeline.py` finishes, launch the interactive dashboard:

```bash
streamlit run app.py
```

### What You Will See in the Dashboard:
- **14-Day Prediction Card**: Color-coded callout badge (`📈 UP`, `📉 DOWN`, or `➡️ FLAT`).
- **Confidence & Probabilities**: Live percentage breakdown for `P(UP)`, `P(FLAT)`, and `P(DOWN)`.
- **Interactive Price Chart**: RAM benchmark prices over time with 7-day and 30-day moving averages and the projected 14-day movement.
- **Top 10 News Headlines**: Most recent headlines with FinBERT sentiment tags (`POSITIVE`, `NEGATIVE`, `NEUTRAL`), domain event tags (`[shortage]`, `[tariff]`, `[demand_surge]`, etc.), date, and publisher source.
- **Top SHAP Features**: Interactive table ranking the key features that drove the prediction.
- **Semiconductor Stock Overview**: Normalized comparative performance chart of Micron, Samsung, SK Hynix, TSMC, and Nvidia.

---

## 📁 Using Your Own RAM Price Data

The target prices are loaded from:
```text
data/raw/ram_prices.csv
```

If missing, the system automatically creates a sample file containing daily benchmark prices with explicit instructions.

### To use your own data:
1. Open `data/raw/ram_prices.csv`.
2. Format your file with two columns:
   ```csv
   date,price
   2024-01-01,3.4500
   2024-01-02,3.4650
   2024-01-03,3.4800
   ```
   - `date`: `YYYY-MM-DD`
   - `price`: Target DRAM/RAM price in USD (e.g. DDR4 8Gb spot price, DDR5 contract price, or RAM kit index).
3. Re-run `python run_pipeline.py`.

---

## 🔬 Methodology & Architecture

### 1. Target Label Definition
For each date $t$, the forward 14-day percentage price change is computed:
$$\text{Return}_{t, t+14} = \frac{P_{t+14} - P_t}{P_t}$$

The discrete classification label is assigned as:
- **`UP` (Class 2)**: $\text{Return}_{t, t+14} > +3.0\%$
- **`DOWN` (Class 0)**: $\text{Return}_{t, t+14} < -3.0\%$
- **`FLAT` (Class 1)**: $-3.0\% \le \text{Return}_{t, t+14} \le +3.0\%$

### 2. Time-Based Walk-Forward Validation (Zero Data Leakage)
In financial time series, random shuffling (`train_test_split(shuffle=True)`) creates lookahead bias and leaks future price information into the past. 

This project strictly employs **Expanding-Window Walk-Forward Validation**:
- **Fold 1**: Train on Days $0 \dots T_1$, Test on Days $T_1 \dots T_2$
- **Fold 2**: Train on Days $0 \dots T_2$, Test on Days $T_2 \dots T_3$
- **Fold 3**: Train on Days $0 \dots T_3$, Test on Days $T_3 \dots T_4$
- **Fold 4**: Train on Days $0 \dots T_4$, Test on Days $T_4 \dots T_5$
- **Fold 5**: Train on Days $0 \dots T_5$, Test on Days $T_5 \dots T_{end}$

All reported test metrics (Accuracy, F1, Confusion Matrix) reflect strictly **out-of-sample forward test windows**.

### 3. Baseline Comparison ("Tomorrow = Today")
We compare LightGBM against the classic financial persistence baseline:
- **Persistence (Flat)**: Assumes prices follow a random walk where price in 14 days equals price today (predicts `FLAT`).
- **Momentum Persistence**: Extrapolates the prior 14-day direction into the next 14 days.

### 4. SHAP (SHapley Additive exPlanations)
To ensure full model interpretability, SHAP values are computed across test observations:
- Identifies the exact contribution of stock price momentum (e.g. `MU_return_14d`, `TSM_return_30d`) vs. news events (`count_supply_cut_roll_7d`, `news_avg_sentiment_7d`).
- Summarized in the terminal output and in the Streamlit UI.

---

## 🛠️ Individual Module Usage

You can also run any module independently:

```bash
# 1. Test target price loading:
python -m src.data.prices

# 2. Test stock data fetching:
python -m src.data.stocks

# 3. Test news collection & GDELT API:
python -m src.data.news

# 4. Test FinBERT sentiment & event tagging:
python -m src.features.sentiment
python -m src.features.events

# 5. Build features dataset:
python -m src.features.build_features

# 6. Train and evaluate model:
python -m src.model.train

# 7. Print CLI forecast summary:
python app.py
```

---

## 📜 Compliance & Safety Note
- **No Web Scraping of E-commerce**: In accordance with system guidelines, this project does not scrape retail sites like Amazon or Flipkart. All signals originate from official financial market APIs (`yfinance`), open news APIs (`GDELT DOC 2.0`), and benchmark commodity spot indices.
