# PhoneWorth – Smartphone Price Predictor

PhoneWorth estimates a **fair price in Indian rupees** for a smartphone from its specs
(chipset, RAM, storage, display, cameras, battery, 5G/NFC and so on). It is trained on
real prices of ~1,000 phones sold in India and shows *why* it arrived at a price.

The app has five pages: **Predict price**, **Deal checker**, **Compare**, **Similar phones**
and **Model report**.

> The earlier version of this repository predicted DRAM price direction. That code is kept
> untouched in `legacy_dram/` and is no longer used.

---

## 1. Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
# macOS only: LightGBM needs the OpenMP runtime
brew install libomp

# 3. Put the dataset in place (see section 2)

# 4. Build features and train the model (about 15 seconds)
python run_pipeline.py

# 5. Open the app
streamlit run app.py
```

Streamlit prints a local URL (normally http://localhost:8501). Open it in a browser.

---

## 2. Data

No data is generated or scraped by this project. It uses one public Kaggle dataset:

* **Smartphone Specifications and Prices in India** by shruti ambekar (CC0, scraped from smartprix.com)
  https://www.kaggle.com/datasets/shrutiambekar/smartphone-specifications-and-prices-in-india
* Download the file `smartphones - smartphones.csv` (1,020 phones) and save it as
  **`data/raw/smartphones.csv`**.

The CSV stores every spec as free text, for example
`"Snapdragon 8 Gen2, Octa Core, 3.2 GHz Processor"` or
`"5000 mAh Battery with 67W Fast Charging"`. The raw columns are also not reliable: when a
phone has no memory-card slot the OS text slides into the `card` column, iPhones have no
battery entry, and so on. The cleaning step therefore joins all spec columns of a row into
one text and pulls each value out with a pattern that only matches that value.

If `data/raw/smartphones.csv` is missing, `run_pipeline.py` stops with a message that
says exactly where to download it.

---

## 3. What the pipeline does

### Step 1 – Load (`src/data/load.py`)
Reads the CSV and checks the expected columns exist. Never creates sample data.

### Step 2 – Clean and build features (`src/features/build.py`)
Parsed from the text: RAM (GB), storage (GB), battery (mAh), fast-charging watts, screen
size (inches), resolution and megapixels, refresh rate (Hz), main rear camera (MP), number
of rear cameras, front camera (MP), 5G / NFC / IR-blaster flags, memory-card slot, OS,
OS version, brand, chipset name, processor family, CPU cores and clock.

* **Chipset tier (1–10)**: a rule-based score for every common chip family
  (Snapdragon 8 Gen 3 = 10, Snapdragon 8 Gen 2 = 9, Dimensity 9200 = 9.5,
  Bionic A17 = 10, Helio G99 = 5, Unisoc T612 = 3, ...). Unknown chips get a blank
  value that the model handles.
* **Outliers removed**: phones under ₹3,000 (26, mostly feature phones) and over
  ₹2,50,000 (2). Clean table: **992 phones, 797 distinct models**.
* **Target**: `log(price)`, so the model's errors are proportional to price.

Output: `data/processed/phones_clean.csv`.

### Step 3 – Train and evaluate (`src/model/train.py`)
* **5-fold GroupKFold grouped by phone model**, so the 6 GB/128 GB and 8 GB/256 GB
  variants of one phone never end up on both sides of a split (no variant leakage).
* Three models are compared on out-of-fold predictions:
  brand-median baseline, Ridge regression and LightGBM.
* Two extra LightGBM **quantile models (10 % and 90 %)** give a likely price range.
* **SHAP** values give feature importance and the per-phone "why this price" list.
* Saved: `models/price_model.pkl` (final model + quantile models) and
  `models/metadata.json` (metrics, importances, dropdown choices).

---

## 4. Results (5-fold GroupKFold, rupee scale)

| Model | MAE (₹) | MAPE | R² | R² (log) | Within 10 % | Within 20 % |
|---|---:|---:|---:|---:|---:|---:|
| Brand median (baseline) | 15,145 | 48.6 % | 0.252 | 0.348 | 13 % | 26 % |
| Ridge | 6,704 | 19.1 % | 0.793 | 0.893 | 36 % | 62 % |
| **LightGBM (final)** | **5,235** | **15.9 %** | **0.877** | **0.922** | **45 %** | **72 %** |

In plain words: on phones the model has never seen, its price is off by about **16 % on
average** (₹5,200), and it explains about 88 % of the variation in price. The baseline
that just guesses the brand's typical price is off by 49 %.

Most important features (mean |SHAP|): chipset tier, storage, screen resolution, brand,
CPU clock, RAM, NFC, fast charging, 5G, battery.

---

## 5. The app (`app.py`)

| Page | What it does |
|---|---|
| **Predict price** | Fill in specs in grouped controls (or start from a phone in the dataset). Shows the fair price, the 10–90 % range and the top 5 reasons with +/− ₹. |
| **Deal checker** | Pick a phone or enter specs plus the shop price. Verdict: *Good deal* (≥ 8 % below fair), *Fair*, or *Overpriced* (≥ 8 % above), with a gauge. |
| **Compare** | 2–3 phones side by side: specs, listed price vs. estimate. |
| **Similar phones** | The 5 closest phones by specs (chipset and memory weighted higher), excluding variants of the same phone. |
| **Model report** | Metrics vs. baseline, predicted-vs-actual scatter, feature importance. |

The design is custom CSS on top of Streamlit (Inter font, one accent colour, rounded
cards) and supports light and dark mode. Theme colours live in `.streamlit/config.toml`.

---

## 6. Project layout

```
.
├── app.py                     # Streamlit app
├── run_pipeline.py            # load -> features -> train -> save
├── requirements.txt
├── .streamlit/config.toml     # light + dark theme
├── data/
│   ├── raw/smartphones.csv        # you download this (see section 2)
│   └── processed/
│       ├── phones_clean.csv       # built by the pipeline
│       └── oof_predictions.csv    # held-out predictions for the report page
├── models/
│   ├── price_model.pkl
│   └── metadata.json
├── src/
│   ├── data/load.py
│   ├── features/build.py
│   └── model/train.py
└── legacy_dram/               # old DRAM project, untouched
```

## 7. Deploy it (free) with Streamlit Community Cloud

The repository already contains everything the server needs: the dataset, the trained
model, `requirements.txt` (pinned versions) and `packages.txt` (the `libgomp1` system
library LightGBM needs on Linux).

1. Push the project to a public GitHub repository (the `venv/` folder is ignored).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository, branch `main`, main file `app.py`.
4. Under *Advanced settings* pick **Python 3.13**, then click **Deploy**.

The first build takes a few minutes. You get a public link like
`https://<your-app-name>.streamlit.app` that you can share anywhere. Every push to
`main` redeploys automatically.

## 8. Limitations

* Prices are as listed on Smartprix at scrape time (early 2023); the market moves.
* The dataset has no launch date, so the model cannot tell a new phone from an old one
  with the same specs (OS version is a weak stand-in).
* Rare brands and chipsets have few examples, so their estimates are less certain;
  the 10–90 % range reflects that.
