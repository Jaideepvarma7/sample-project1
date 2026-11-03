"""Module for walk-forward time-series cross-validation.

Implements expanding-window walk-forward validation:
- Strictly chronological order (NO data shuffling to eliminate lookahead leakage)
- Fold i trains on past data [0 : t_i] and evaluates on out-of-sample future slice [t_i : t_{i+1}]
- Computes out-of-fold validation metrics across all chronological periods
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.model.classifier import get_classifier
from src.model.evaluate import compute_classification_metrics

logger = logging.getLogger(__name__)


def generate_time_series_splits(
    n_samples: int,
    n_splits: int = 5,
    min_train_ratio: float = 0.40,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window indices for walk-forward validation.

    Args:
        n_samples: Total number of sequential samples.
        n_splits: Number of time folds (default: 5).
        min_train_ratio: Minimum fraction of data for the first training window.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    min_train_samples = int(n_samples * min_train_ratio)
    test_block_size = (n_samples - min_train_samples) // n_splits

    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_splits):
        train_end = min_train_samples + i * test_block_size
        test_start = train_end
        test_end = (
            min_train_samples + (i + 1) * test_block_size
            if i < n_splits - 1
            else n_samples
        )

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


def run_walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Execute walk-forward cross-validation on time-ordered data.

    Args:
        X: Feature DataFrame (sorted chronologically).
        y: Target Series (0=DOWN, 1=FLAT, 2=UP).
        n_splits: Number of time slices.

    Returns:
        Dict containing out-of-fold metrics, per-fold details, and concatenated predictions.
    """
    n_samples = len(X)
    y_arr = np.asarray(y, dtype=int)
    splits = generate_time_series_splits(n_samples=n_samples, n_splits=n_splits)

    oof_preds = np.zeros(n_samples, dtype=int)
    oof_mask = np.zeros(n_samples, dtype=bool)
    oof_probs = np.zeros((n_samples, 3), dtype=float)

    fold_details: List[Dict[str, Any]] = []

    logger.info("Starting %d-fold walk-forward validation...", n_splits)

    for fold_num, (train_idx, test_idx) in enumerate(splits, 1):
        X_train, y_train = X.iloc[train_idx], y_arr[train_idx]
        X_test, y_test = X.iloc[test_idx], y_arr[test_idx]

        model = get_classifier()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        oof_preds[test_idx] = preds
        oof_probs[test_idx] = probs
        oof_mask[test_idx] = True

        fold_metrics = compute_classification_metrics(y_test, preds)
        fold_details.append({
            "fold": fold_num,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "accuracy": fold_metrics["accuracy"],
            "macro_f1": fold_metrics["macro_f1"],
            "weighted_f1": fold_metrics["weighted_f1"],
        })
        logger.info(
            "Fold %d/%d: Train=%d, Test=%d | Acc=%.2f%%, Macro F1=%.3f",
            fold_num,
            n_splits,
            len(train_idx),
            len(test_idx),
            fold_metrics["accuracy"] * 100,
            fold_metrics["macro_f1"],
        )

    # Compute overall Out-Of-Fold (OOF) metrics
    y_oof_true = y_arr[oof_mask]
    y_oof_pred = oof_preds[oof_mask]

    overall_metrics = compute_classification_metrics(y_oof_true, y_oof_pred)

    return {
        "overall_metrics": overall_metrics,
        "fold_details": fold_details,
        "oof_preds": y_oof_pred,
        "oof_true": y_oof_true,
        "oof_probs": oof_probs[oof_mask],   # shape (n_oof, 3) — added for confidence filtering
        "oof_indices": np.where(oof_mask)[0],
        "n_splits": n_splits,
    }
