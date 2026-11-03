"""Master model training, validation, and evaluation pipeline.

Coordinates:
1. Target labeling (>+3% UP, <-3% DOWN, else FLAT)
2. Walk-forward cross validation (no future leakage, strict time order)
3. Benchmark against 'tomorrow = today' baseline
4. Full model training & SHAP feature importance analysis
5. Model artifact serialization (models/lgbm_model.pkl, models/metadata.json)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.model.classifier import get_classifier
from src.model.evaluate import (
    compute_classification_metrics,
    compute_confidence_filtered_metrics,
    compute_feature_importances,
    evaluate_persistence_baseline,
    generate_verdict,
    print_evaluation_summary,
)
from src.model.labeler import (
    CLASS_NAMES,
    LABEL_MAP,
    create_14day_target,
    prepare_features_and_target,
)
from src.model.walk_forward import run_walk_forward_validation

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "lgbm_model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


def train_and_evaluate(
    dataset_path: str = "data/processed/final_dataset.csv",
    model_save_path: Path | str = MODEL_PATH,
    metadata_save_path: Path | str = METADATA_PATH,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Execute end-to-end model training, walk-forward evaluation, and serialization.

    Args:
        dataset_path: Path to engineered final_dataset.csv.
        model_save_path: Destination path for trained model pickle.
        metadata_save_path: Destination path for JSON metadata.
        n_splits: Number of walk-forward time splits.

    Returns:
        Dict with evaluation metrics, baseline metrics, and feature importances.
    """
    model_save_path = Path(model_save_path)
    metadata_save_path = Path(metadata_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s...", dataset_path)
    raw_df = pd.read_csv(dataset_path)

    # 1. Generate 14-day forward return labels
    labeled_df = create_14day_target(raw_df, price_col="price", horizon=14, threshold=0.03)

    # 2. Extract feature matrix X and target y (drops rows with unobserved future labels)
    X, y, clean_df = prepare_features_and_target(labeled_df)

    # 3. Walk-Forward Validation (Strict Time-Series Splits, no shuffling)
    wf_results = run_walk_forward_validation(X, y, n_splits=n_splits)
    oof_metrics = wf_results["overall_metrics"]

    # 4. Baseline Evaluation — flat, majority class, and momentum
    #    All evaluated on the exact same out-of-fold samples (no leakage)
    oof_indices = wf_results["oof_indices"]
    oof_df_slice = clean_df.iloc[oof_indices]
    oof_y_true = wf_results["oof_true"]
    oof_y_pred = wf_results["oof_preds"]
    oof_probs  = wf_results["oof_probs"]      # shape (n_oof, 3) — columns: DOWN, FLAT, UP

    baseline_results = evaluate_persistence_baseline(oof_df_slice, oof_y_true)

    # 4b. Confidence-filtered metrics: only call UP/DOWN when P > 0.60, else say FLAT
    confidence_metrics = compute_confidence_filtered_metrics(
        y_true=oof_y_true,
        y_pred=oof_y_pred,
        y_proba=oof_probs,
        confidence_threshold=0.60,
    )

    # 5. Train Final Model on all available historical labeled data
    logger.info("Training final production LightGBM model on all %d samples...", len(X))
    final_model = get_classifier()
    final_model.fit(X, y)

    # 6. Compute Feature Importances (SHAP / Explainer)
    top_features = compute_feature_importances(final_model, X, top_n=15)

    # 7. Print Terminal Summary (includes majority baseline + confidence filter + verdict)
    print_evaluation_summary(
        model_metrics=oof_metrics,
        baseline_metrics=baseline_results,
        top_features=top_features,
        confidence_metrics=confidence_metrics,
        title="WALK-FORWARD MODEL VALIDATION SUMMARY",
    )

    # 7b. Print plain-English verdict
    verdict_text = generate_verdict(oof_metrics, baseline_results, confidence_metrics)
    print(verdict_text)

    # 8. Save Model and Metadata
    with open(model_save_path, "wb") as f:
        pickle.dump(final_model, f)
    logger.info("Saved trained model to %s", model_save_path)

    metadata = {
        "model_type": "LightGBM Classifier",
        "target_horizon_days": 14,
        "classes": CLASS_NAMES,
        "feature_names": list(X.columns),
        "n_samples": len(X),
        "n_features": len(X.columns),
        "walk_forward_metrics": oof_metrics,
        "baseline_metrics": baseline_results,
        "confidence_metrics": {
            "threshold": confidence_metrics["threshold"],
            "n_confident": confidence_metrics["n_confident"],
            "n_total": confidence_metrics["n_total"],
            "pct_confident": confidence_metrics["pct_confident"],
            "accuracy_confident_calls": confidence_metrics["metrics_confident"].get("accuracy"),
        },
        "fold_details": wf_results["fold_details"],
        "top_features": top_features,
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to %s", metadata_save_path)

    return {
        "model": final_model,
        "metadata": metadata,
        "oof_metrics": oof_metrics,
        "baseline_results": baseline_results,
        "confidence_metrics": confidence_metrics,
        "top_features": top_features,
    }


def predict_latest_forecast(
    dataset_path: str = "data/processed/final_dataset.csv",
    model_path: Path | str = MODEL_PATH,
    metadata_path: Path | str = METADATA_PATH,
) -> Dict[str, Any]:
    """Generate the latest 14-day price prediction using the most recent observation.

    Args:
        dataset_path: Path to engineered final_dataset.csv.
        model_path: Path to serialized model.
        metadata_path: Path to metadata JSON.

    Returns:
        Dict with prediction: 'UP', 'DOWN', or 'FLAT', confidence, probabilities,
        and latest market indicators.
    """
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model artifacts not found. Please run train_and_evaluate() first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_names = metadata["feature_names"]
    df = pd.read_csv(dataset_path)

    latest_row = df.iloc[[-1]]
    latest_date = str(latest_row["date"].values[0])
    latest_price = float(latest_row["price"].values[0])

    X_latest = latest_row[feature_names]

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_latest)[0]
    else:
        probs = np.array([0.33, 0.33, 0.34])

    pred_class_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_class_idx]
    confidence = float(probs[pred_class_idx])

    prob_dict = {
        "DOWN": round(float(probs[0]), 4),
        "FLAT": round(float(probs[1]), 4),
        "UP": round(float(probs[2]), 4),
    }

    return {
        "as_of_date": latest_date,
        "current_price": round(latest_price, 4),
        "prediction": pred_label,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "horizon_days": 14,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = train_and_evaluate()
    latest = predict_latest_forecast()
    print("\nLATEST 14-DAY FORECAST:")
    print(json.dumps(latest, indent=2))
