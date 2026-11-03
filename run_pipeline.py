"""Run the whole PhoneWorth pipeline with one command:

    python run_pipeline.py

Steps:
  1. Load   data/raw/smartphones.csv            (src/data/load.py)
  2. Clean  -> data/processed/phones_clean.csv  (src/features/build.py)
  3. Train  5-fold GroupKFold, compare models, fit final LightGBM + quantiles,
            SHAP importance                     (src/model/train.py)
  4. Save   models/price_model.pkl + models/metadata.json

Then start the app with:  streamlit run app.py
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    t0 = time.time()

    print("=" * 64)
    print("STEP 1  Load raw data")
    print("=" * 64)
    from src.data.load import DatasetNotFoundError, load_raw

    try:
        raw = load_raw()
    except DatasetNotFoundError as e:
        print(e)
        return 1
    print(f"Loaded {len(raw)} rows, {len(raw.columns)} columns\n")

    print("=" * 64)
    print("STEP 2  Clean + build features")
    print("=" * 64)
    from src.features.build import build_features, save_clean

    clean = build_features(raw)
    removed = dict(clean.attrs.get("removed", {}))
    path = save_clean(clean)
    print(f"Saved {path}\n")

    print("=" * 64)
    print("STEP 3  Train + evaluate + save model")
    print("=" * 64)
    from src.model.train import run

    meta = run(clean, removed=removed)

    h = meta["headline"]
    print("\n" + "=" * 64)
    print(f"DONE in {time.time() - t0:.1f}s  |  LightGBM: MAE Rs {h['mae_inr']:,}  MAPE {h['mape_pct']}%  R2 {h['r2']}")
    print("Start the app with:  streamlit run app.py")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
