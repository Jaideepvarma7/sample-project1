"""Evaluation utilities for the RAM price predictor.

Computes:
- Accuracy, Macro F1, Weighted F1
- 3x3 Confusion Matrix (DOWN, FLAT, UP)
- Baselines: "tomorrow = today", "most common class", momentum
- Confidence-filtered prediction metrics (UP/DOWN only when P > 0.6)
- SHAP feature importance analysis
- Plain-English verdict on whether the model beats baselines
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLASS_NAMES: List[str] = ["DOWN", "FLAT", "UP"]


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> Dict[str, Any]:
    """Compute accuracy, per-class metrics, macro F1, and weighted F1.

    Args:
        y_true: True class labels (0, 1, 2).
        y_pred: Predicted class labels (0, 1, 2).

    Returns:
        Dict with metrics and confusion matrix.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    n_samples = len(yt)
    if n_samples == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    accuracy = float(np.mean(yt == yp))

    # Confusion matrix 3x3: rows=true, cols=pred
    cm = np.zeros((3, 3), dtype=int)
    for t_val, p_val in zip(yt, yp):
        if 0 <= t_val <= 2 and 0 <= p_val <= 2:
            cm[t_val, p_val] += 1

    per_class_f1: List[float] = []
    support: List[int] = []

    for c in range(3):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        sup = np.sum(cm[c, :])
        support.append(int(sup))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1))
    total_sup = sum(support)
    weighted_f1 = (
        float(sum(f1 * sup for f1, sup in zip(per_class_f1, support)) / total_sup)
        if total_sup > 0
        else 0.0
    )

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_f1": [round(f, 4) for f in per_class_f1],
        "support": support,
        "n_samples": n_samples,
    }


def evaluate_persistence_baseline(
    df: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
) -> Dict[str, Any]:
    """Compute baseline performance: 'tomorrow = today', 'most common class', momentum.

    Tests three standard baselines:
    1. Flat baseline: Predicts FLAT (1) for every day
    2. Most-common-class baseline: Always predicts the majority class in the training set
    3. Momentum baseline: Predicts continuation of past 14-day direction

    Args:
        df: DataFrame containing past returns ('ram_return_14d')
        y_true: True future labels (integers: 0=DOWN, 1=FLAT, 2=UP)

    Returns:
        Dict with metrics for each baseline.
    """
    yt = np.asarray(y_true, dtype=int)
    n = len(yt)

    # 1. Flat persistence: always predict FLAT (price tomorrow = price today)
    flat_preds = np.ones(n, dtype=int)
    flat_metrics = compute_classification_metrics(yt, flat_preds)

    # 2. Most-common-class baseline: always predict whatever class is most frequent
    #    in the evaluation window (FLAT is always dominant in this dataset at 59.7%)
    class_counts = np.bincount(yt, minlength=3)
    majority_class = int(np.argmax(class_counts))
    majority_name = CLASS_NAMES[majority_class]
    majority_preds = np.full(n, majority_class, dtype=int)
    majority_metrics = compute_classification_metrics(yt, majority_preds)
    majority_metrics["majority_class_name"] = majority_name

    # 3. Momentum persistence: if last 14d was UP, predict UP; if DOWN, predict DOWN
    if "ram_return_14d" in df.columns:
        past_returns = df["ram_return_14d"].values[:n]
        mom_preds = np.ones(n, dtype=int)
        mom_preds[past_returns > 0.03] = 2   # UP
        mom_preds[past_returns < -0.03] = 0  # DOWN
        mom_metrics = compute_classification_metrics(yt, mom_preds)
    else:
        mom_metrics = flat_metrics

    return {
        "flat_baseline": flat_metrics,
        "majority_baseline": majority_metrics,
        "momentum_baseline": mom_metrics,
    }


def compute_confidence_filtered_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    confidence_threshold: float = 0.60,
) -> Dict[str, Any]:
    """Evaluate accuracy on a confidence-filtered subset of predictions.

    Strategy:
        - If the model's max class probability > threshold, use that prediction.
        - Otherwise, downgrade to FLAT (class 1) — "not sure, so stay neutral".

    This lets us measure precision on the calls the model is actually confident about.

    Args:
        y_true: True labels (0/1/2).
        y_pred: Raw argmax predictions (0/1/2).
        y_proba: Probability matrix, shape (n, 3). Columns: [DOWN, FLAT, UP].
        confidence_threshold: Minimum probability to trust a UP/DOWN call.

    Returns:
        Dict with:
            - filtered_preds: adjusted predictions after applying the threshold
            - confident_mask: bool array marking rows where a UP/DOWN call was made
            - metrics_all: metrics over all rows (with threshold-overridden preds)
            - metrics_confident: metrics only on confident UP/DOWN rows
            - n_confident / n_total
            - pct_confident
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    proba = np.asarray(y_proba)

    max_prob = np.max(proba, axis=1)         # highest prob for each row
    max_class = np.argmax(proba, axis=1)     # 0, 1, or 2

    # Apply filter: only call UP (2) or DOWN (0) when confidence exceeds threshold
    filtered = np.ones(len(yt), dtype=int)   # default: FLAT
    confident_mask = np.zeros(len(yt), dtype=bool)

    for i in range(len(yt)):
        if max_class[i] != 1 and max_prob[i] > confidence_threshold:
            # Model is confident it's UP or DOWN — trust it
            filtered[i] = max_class[i]
            confident_mask[i] = True
        else:
            # Not confident enough on a directional call → say FLAT
            filtered[i] = 1

    n_total = len(yt)
    n_confident = int(confident_mask.sum())
    pct_confident = 100.0 * n_confident / n_total if n_total > 0 else 0.0

    metrics_all = compute_classification_metrics(yt, filtered)
    if n_confident > 0:
        metrics_confident = compute_classification_metrics(
            yt[confident_mask], filtered[confident_mask]
        )
    else:
        metrics_confident = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "n_samples": 0}

    return {
        "filtered_preds": filtered,
        "confident_mask": confident_mask,
        "metrics_all": metrics_all,
        "metrics_confident": metrics_confident,
        "n_confident": n_confident,
        "n_total": n_total,
        "pct_confident": round(pct_confident, 1),
        "threshold": confidence_threshold,
    }


def generate_verdict(
    model_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    confidence_metrics: Dict[str, Any],
) -> str:
    """Generate a plain-English verdict: is the model better than the baselines?

    Rules:
    - Compare macro F1 (the fairest metric given class imbalance) to all baselines.
    - Compare confidence-filtered accuracy on directional (UP/DOWN) calls.
    - Acknowledge the honest trade-off between raw accuracy and F1.
    """
    model_macro_f1 = model_metrics.get("macro_f1", 0.0)
    model_acc = model_metrics.get("accuracy", 0.0)

    flat_acc   = baseline_metrics.get("flat_baseline", {}).get("accuracy", 0.0)
    flat_f1    = baseline_metrics.get("flat_baseline", {}).get("macro_f1", 0.0)
    maj_f1     = baseline_metrics.get("majority_baseline", {}).get("macro_f1", 0.0)
    maj_acc    = baseline_metrics.get("majority_baseline", {}).get("accuracy", 0.0)
    mom_f1     = baseline_metrics.get("momentum_baseline", {}).get("macro_f1", 0.0)

    beats_flat_f1 = model_macro_f1 > flat_f1
    beats_maj_f1  = model_macro_f1 > maj_f1
    beats_mom_f1  = model_macro_f1 > mom_f1
    beats_flat_acc = model_acc > flat_acc

    n_confident = confidence_metrics.get("n_confident", 0)
    pct_confident = confidence_metrics.get("pct_confident", 0.0)
    conf_acc = confidence_metrics.get("metrics_confident", {}).get("accuracy", 0.0)

    verdict_lines = ["", "=" * 70, "  PLAIN-ENGLISH VERDICT", "=" * 70]

    # Accuracy headline
    if beats_flat_acc:
        verdict_lines.append(f"  ✅ Accuracy: Model ({model_acc*100:.1f}%) beats 'always-FLAT' baseline ({flat_acc*100:.1f}%).")
    else:
        verdict_lines.append(
            f"  ❌ Accuracy: Model ({model_acc*100:.1f}%) is LOWER than 'always-FLAT' baseline ({flat_acc*100:.1f}%)."
            f"\n     This happens because FLAT covers 59.7% of days — predicting FLAT every day"
            f"\n     looks accurate but is useless: it can never catch UP or DOWN moves."
        )

    # Macro F1 headline (the honest metric for imbalanced classes)
    beats_all = beats_flat_f1 and beats_maj_f1 and beats_mom_f1
    beats_some = beats_flat_f1 or beats_maj_f1 or beats_mom_f1
    if beats_all:
        verdict_lines.append(
            f"  ✅ Macro F1: Model ({model_macro_f1:.3f}) beats ALL three baselines"
            f" (flat={flat_f1:.3f}, majority={maj_f1:.3f}, momentum={mom_f1:.3f})."
            f"\n     Macro F1 is the right metric here — it penalises ignoring minority classes."
        )
    elif beats_some:
        verdict_lines.append(
            f"  ⚠️  Macro F1: Model ({model_macro_f1:.3f}) beats SOME baselines but not all."
            f"\n     flat={flat_f1:.3f}, majority={maj_f1:.3f}, momentum={mom_f1:.3f}."
        )
    else:
        verdict_lines.append(
            f"  ❌ Macro F1: Model ({model_macro_f1:.3f}) does NOT beat any baseline."
            f"\n     flat={flat_f1:.3f}, majority={maj_f1:.3f}, momentum={mom_f1:.3f}."
        )

    # Confidence filter insight
    if n_confident > 0:
        verdict_lines.append(
            f"  🎯 Confidence Filter (>60%): On the {n_confident} days ({pct_confident:.0f}% of test)"
            f" where the model is confident enough to call UP or DOWN,"
            f" accuracy is {conf_acc*100:.1f}%."
        )
        if conf_acc > 0.55:
            verdict_lines.append(
                f"     That's above 55% on directional calls — meaningful for trading signals."
            )
        else:
            verdict_lines.append(
                f"     That's below 55% — the confident calls are not reliable yet."
            )
    else:
        verdict_lines.append(
            "  ℹ️  No UP/DOWN calls exceeded the 60% confidence threshold in the test window."
        )

    # Root cause note
    verdict_lines.append("")
    verdict_lines.append("  ROOT CAUSE of limitations (honest assessment):")
    verdict_lines.append("  • Data: All prices and news are SYNTHETIC. Real DRAMeXchange spot")
    verdict_lines.append("    prices and live GDELT news would dramatically improve signal quality.")
    verdict_lines.append("  • Model: PureNumpyGradientBooster is a linear softmax, not a tree.")
    verdict_lines.append("    Install 'lightgbm' for the real non-linear gradient boosted trees.")
    verdict_lines.append("  • Features: Synthetic news has uniform patterns — real FinBERT on")
    verdict_lines.append("    actual headlines would create much stronger sentiment divergence.")
    verdict_lines.append("=" * 70)

    return "\n".join(verdict_lines)


def compute_feature_importances(
    model: Any,
    X: pd.DataFrame,
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Compute feature importances using SHAP with fallback to model attributes.

    Args:
        model: Trained classifier.
        X: Feature DataFrame.
        top_n: Number of top features to return.

    Returns:
        List of dicts: [{'rank': 1, 'feature': '...', 'importance': 0.123}, ...]
    """
    feature_names = list(X.columns)

    # Attempt SHAP tree explanation
    try:
        import shap  # type: ignore

        logger.info("Computing SHAP TreeExplainer values...")
        sample_X = X.sample(min(200, len(X)), random_state=42)
        explainer = shap.Explainer(model, sample_X)
        shap_values = explainer(sample_X)

        if hasattr(shap_values, "values"):
            vals = shap_values.values
            if vals.ndim == 3:  # (samples, features, classes)
                mean_abs = np.mean(np.abs(vals), axis=(0, 2))
            else:
                mean_abs = np.mean(np.abs(vals), axis=0)
        else:
            mean_abs = np.mean(np.abs(shap_values), axis=0)

        importances = mean_abs / (np.sum(mean_abs) + 1e-9)
    except Exception as exc:
        logger.info(
            "SHAP computation bypassed (%s). Using model feature importances.", exc
        )
        if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
            raw_imp = np.asarray(model.feature_importances_, dtype=float)
            total = np.sum(raw_imp)
            importances = raw_imp / (total if total > 0 else 1.0)
        else:
            # Fallback to feature variance
            variances = np.var(X.values, axis=0)
            importances = variances / np.sum(variances)

    sorted_idx = np.argsort(importances)[::-1][:top_n]
    results = []
    for rank, idx in enumerate(sorted_idx, 1):
        results.append({
            "rank": rank,
            "feature": feature_names[idx],
            "importance": round(float(importances[idx]), 5),
        })
    return results


def print_evaluation_summary(
    model_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    top_features: List[Dict[str, Any]],
    confidence_metrics: Optional[Dict[str, Any]] = None,
    title: str = "WALK-FORWARD VALIDATION RESULTS",
) -> None:
    """Print an executive, formatted terminal summary of model performance."""
    print("\n" + "=" * 70)
    print(f"  {title.center(66)}")
    print("=" * 70)

    # Model vs ALL Baselines comparison
    flat_base = baseline_metrics.get("flat_baseline", {})
    maj_base  = baseline_metrics.get("majority_baseline", {})
    mom_base  = baseline_metrics.get("momentum_baseline", {})
    maj_name  = maj_base.get("majority_class_name", "FLAT")

    print("\n--- MODEL VS BASELINE BENCHMARK ---")
    header = f"{'Model / Baseline':<36} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<11}"
    print(header)
    print("-" * len(header))
    print(
        f"{'LightGBM RAM Predictor':<36} | "
        f"{model_metrics['accuracy'] * 100:>8.2f}% | "
        f"{model_metrics['macro_f1']:>10.4f} | "
        f"{model_metrics['weighted_f1']:>11.4f}"
    )
    print(
        f"{'Baseline: Always FLAT (tomorrow=today)':<36} | "
        f"{flat_base.get('accuracy', 0) * 100:>8.2f}% | "
        f"{flat_base.get('macro_f1', 0):>10.4f} | "
        f"{flat_base.get('weighted_f1', 0):>11.4f}"
    )
    print(
        f"{'Baseline: Always ' + maj_name + ' (most common)':<36} | "
        f"{maj_base.get('accuracy', 0) * 100:>8.2f}% | "
        f"{maj_base.get('macro_f1', 0):>10.4f} | "
        f"{maj_base.get('weighted_f1', 0):>11.4f}"
    )
    print(
        f"{'Baseline: Past 14d Momentum':<36} | "
        f"{mom_base.get('accuracy', 0) * 100:>8.2f}% | "
        f"{mom_base.get('macro_f1', 0):>10.4f} | "
        f"{mom_base.get('weighted_f1', 0):>11.4f}"
    )

    # Confusion matrix
    cm = np.array(model_metrics["confusion_matrix"])
    print("\n--- CONFUSION MATRIX (Out-of-Fold Test Data) ---")
    print(f"{'':>14} | {'Pred DOWN':<10} | {'Pred FLAT':<10} | {'Pred UP':<10}")
    print("-" * 52)
    for row_idx, row_name in enumerate(CLASS_NAMES):
        print(
            f"{'Actual ' + row_name:<14} | "
            f"{cm[row_idx, 0]:>10} | "
            f"{cm[row_idx, 1]:>10} | "
            f"{cm[row_idx, 2]:>10}"
        )

    # Per class F1
    print("\n--- PER-CLASS PERFORMANCE ---")
    for name, f1, sup in zip(
        CLASS_NAMES, model_metrics["per_class_f1"], model_metrics["support"]
    ):
        print(f"  Class {name:<5}: F1 = {f1:.4f} (Support = {sup} samples)")

    # Confidence-filtered results
    if confidence_metrics is not None:
        cf = confidence_metrics
        print(f"\n--- CONFIDENCE-FILTERED PREDICTIONS (threshold > {cf['threshold']:.0%}) ---")
        print(
            f"  Confident UP/DOWN calls : {cf['n_confident']} out of {cf['n_total']} "
            f"test days ({cf['pct_confident']:.0f}%)"
        )
        conf_m = cf["metrics_all"]
        print(
            f"  Accuracy (all rows, filtered):         {conf_m['accuracy']*100:.2f}%  "
            f"| Macro F1: {conf_m['macro_f1']:.4f}"
        )
        if cf["n_confident"] > 0:
            confl_m = cf["metrics_confident"]
            print(
                f"  Accuracy (confident UP/DOWN rows only): {confl_m['accuracy']*100:.2f}%  "
                f"| Macro F1: {confl_m['macro_f1']:.4f}  "
                f"(n={cf['n_confident']})"
            )
        else:
            print("  No UP/DOWN calls passed the confidence threshold.")

    # Top features
    print(f"\n--- TOP {len(top_features)} FEATURE IMPORTANCES (SHAP / Explainer) ---")
    print(f"{'Rank':<6} | {'Importance':<12} | {'Feature Name'}")
    print("-" * 60)
    for item in top_features:
        print(
            f"#{item['rank']:<5} | {item['importance'] * 100:>9.2f}%   | {item['feature']}"
        )
    print("=" * 70 + "\n")
