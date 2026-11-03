"""Train and evaluate the smartphone price model.

What happens here, in order:
1. Load the clean feature table (data/processed/phones_clean.csv).
2. 5-fold GroupKFold cross-validation, grouped by the base phone name, so
   all variants of one phone (e.g. 6GB/128GB and 8GB/256GB) stay in the same
   fold. Without this the model could "cheat" by memorising a sibling variant.
3. Compare three models on out-of-fold predictions:
     - brand-median baseline (predicts the median price of the brand)
     - Ridge regression (linear, one-hot brands)
     - LightGBM (gradient boosted trees)
   Metrics: MAE in rupees, MAPE in %, R-squared, all on the rupee scale.
4. Fit the final LightGBM on all data, plus two quantile models (10% / 90%)
   that give a low/high price range.
5. SHAP feature importance.
6. Save models/price_model.pkl (everything the app needs) and
   models/metadata.json (metrics, importances, dropdown choices).

The target is log(price); predictions are converted back with exp().
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features.build import (
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    NUMERIC_FEATURES,
    FEATURES,
    load_clean,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "price_model.pkl"
META_PATH = MODEL_DIR / "metadata.json"
OOF_PATH = PROJECT_ROOT / "data" / "processed" / "oof_predictions.csv"

RANDOM_STATE = 42
N_FOLDS = 5

LGBM_PARAMS = dict(
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=8,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    verbose=-1,
)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class BrandMedianBaseline:
    """Predict the median log-price of the phone's brand (global median if unseen)."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_ = float(np.median(y))
        self.by_brand_ = pd.Series(y.values, index=X["brand"].values).groupby(level=0).median().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.by_brand_.get(b, self.global_) for b in X["brand"]])


def make_ridge() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", Ridge(alpha=3.0))])


def make_lgbm(**overrides) -> Pipeline:
    """LightGBM pipeline. Categoricals are ordinal-encoded and flagged as
    categorical for LightGBM, so it can split on brand etc. natively."""
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)
    params = {**LGBM_PARAMS, **overrides}
    return Pipeline([("pre", pre), ("model", lgb.LGBMRegressor(**params))])


def fit_lgbm(pipe: Pipeline, X: pd.DataFrame, y) -> Pipeline:
    cat_idx = list(range(len(NUMERIC_FEATURES), len(FEATURES)))
    pipe.fit(X, y, model__categorical_feature=cat_idx)
    return pipe


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def metrics_inr(price: np.ndarray, pred_log: np.ndarray) -> dict:
    pred = np.exp(pred_log)
    err = pred - price
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((price - price.mean()) ** 2))
    log_true = np.log(price)
    ss_res_log = float(np.sum((pred_log - log_true) ** 2))
    ss_tot_log = float(np.sum((log_true - log_true.mean()) ** 2))
    return {
        "MAE_inr": float(np.mean(np.abs(err))),
        "MAPE_pct": float(np.mean(np.abs(err) / price) * 100),
        "R2": 1 - ss_res / ss_tot,
        "R2_log": 1 - ss_res_log / ss_tot_log,
        "within_10pct": float(np.mean(np.abs(err) / price <= 0.10) * 100),
        "within_20pct": float(np.mean(np.abs(err) / price <= 0.20) * 100),
    }


def format_table(results: dict[str, dict]) -> str:
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "MAE (Rs)": f"{m['MAE_inr']:,.0f}",
            "MAPE %": f"{m['MAPE_pct']:.1f}",
            "R2": f"{m['R2']:.3f}",
            "R2 (log)": f"{m['R2_log']:.3f}",
            "within 10%": f"{m['within_10pct']:.0f}%",
            "within 20%": f"{m['within_20pct']:.0f}%",
        })
    return pd.DataFrame(rows).to_string(index=False)


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def cross_validate(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    X, y, groups = df[FEATURES], df["log_price"], df["base_model"]
    oof = {name: np.zeros(len(df)) for name in ["Brand median", "Ridge", "LightGBM"]}
    gkf = GroupKFold(n_splits=N_FOLDS)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y.iloc[tr]
        oof["Brand median"][te] = BrandMedianBaseline().fit(Xtr, ytr).predict(Xte)
        oof["Ridge"][te] = make_ridge().fit(Xtr, ytr).predict(Xte)
        oof["LightGBM"][te] = fit_lgbm(make_lgbm(), Xtr, ytr).predict(Xte)
        print(f"  fold {fold}/{N_FOLDS} done ({len(te)} test phones)")

    price = df["price"].values
    results = {name: metrics_inr(price, p) for name, p in oof.items()}
    oof_df = df[["model", "base_model", "brand", "price"]].copy()
    for name, p in oof.items():
        oof_df[f"pred_{name.lower().replace(' ', '_')}"] = np.exp(p)
    return results, oof_df


# --------------------------------------------------------------------------
# Final fit, SHAP, save
# --------------------------------------------------------------------------

def shap_importance(pipe: Pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    import shap

    Xt = pipe["pre"].transform(X)
    explainer = shap.TreeExplainer(pipe["model"])
    values = explainer.shap_values(Xt)
    imp = pd.DataFrame({
        "feature": FEATURES,
        "label": [FEATURE_LABELS.get(f, f) for f in FEATURES],
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return imp, float(np.ravel(explainer.expected_value)[0])


def train_and_save(df: pd.DataFrame, results: dict, oof_df: pd.DataFrame, removed: dict | None = None) -> dict:
    X, y = df[FEATURES], df["log_price"]

    print("Fitting final LightGBM on all phones...")
    model = fit_lgbm(make_lgbm(), X, y)
    print("Fitting 10% / 90% quantile models for the price range...")
    q10 = fit_lgbm(make_lgbm(objective="quantile", alpha=0.10), X, y)
    q90 = fit_lgbm(make_lgbm(objective="quantile", alpha=0.90), X, y)

    print("Computing SHAP importance...")
    imp, expected_value = shap_importance(model, X)

    # Dropdown choices for the app, taken from the real data.
    chip_table = (df.dropna(subset=["chipset"])
                    .groupby("chipset")
                    .agg(tier=("chip_tier", "first"), processor_brand=("processor_brand", "first"), n=("chipset", "size"))
                    .reset_index().sort_values(["tier", "n"], ascending=[False, False]))
    chip_table = chip_table[chip_table["tier"].notna()]

    bundle = {
        "model": model,
        "q10": q10,
        "q90": q90,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "shap_expected_value": expected_value,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(OOF_PATH, index=False)

    best = results["LightGBM"]
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_phones": int(len(df)),
        "n_distinct_models": int(df["base_model"].nunique()),
        "cv": {"method": "GroupKFold", "folds": N_FOLDS, "group_by": "base_model"},
        "metrics": results,
        "headline": {
            "mape_pct": round(best["MAPE_pct"], 1),
            "mae_inr": round(best["MAE_inr"]),
            "r2": round(best["R2"], 3),
        },
        "removed_rows": removed or {},
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "shap_importance": imp.to_dict(orient="records"),
        "lgbm_params": LGBM_PARAMS,
        "choices": {
            "brands": sorted(df["brand"].unique().tolist()),
            "processor_brands": sorted(df["processor_brand"].unique().tolist()),
            "os": sorted(df["os"].unique().tolist()),
            "chipsets": chip_table.to_dict(orient="records"),
        },
        "price_range_inr": [float(df["price"].min()), float(df["price"].max())],
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved {MODEL_PATH}")
    print(f"Saved {META_PATH}")
    return metadata


def run(df: pd.DataFrame | None = None, removed: dict | None = None) -> dict:
    t0 = time.time()
    if df is None:
        df = load_clean()
    print(f"Training on {len(df)} phones ({df['base_model'].nunique()} distinct models)")
    print(f"Running {N_FOLDS}-fold GroupKFold cross-validation...")
    results, oof_df = cross_validate(df)
    print("\nCross-validated results (out-of-fold, rupee scale):")
    print(format_table(results))
    print()
    meta = train_and_save(df, results, oof_df, removed)
    print("\nTop 10 features by mean |SHAP| (log-price units):")
    for r in meta["shap_importance"][:10]:
        print(f"  {r['label']:<22} {r['mean_abs_shap']:.3f}")
    print(f"\nDone in {time.time() - t0:.1f}s")
    return meta


if __name__ == "__main__":
    run()
