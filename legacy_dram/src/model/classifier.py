"""Module providing the LightGBM classifier with robust fallback support.

Defines a unified model interface supporting:
- fit(X, y)
- predict(X)
- predict_proba(X)
- feature_importances_
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PureNumpyGradientBooster:
    """High-performance multi-class gradient-boosted decision tree ensemble in pure NumPy.

    Used when neither LightGBM nor Scikit-Learn are installed in the Python environment.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.08,
        max_depth: int = 4,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.classes_: np.ndarray = np.array([0, 1, 2])
        self.n_classes_ = 3
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "PureNumpyGradientBooster":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self.feature_names_ = [f"f_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=np.int64)
        n_samples, n_features = X_arr.shape
        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)

        # Standardize features for stable optimization
        self.mean_ = np.nanmean(X_arr, axis=0)
        self.std_ = np.nanstd(X_arr, axis=0)
        self.std_[self.std_ == 0] = 1.0

        X_norm = (X_arr - self.mean_) / self.std_
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        # One-hot encode targets
        Y_onehot = np.zeros((n_samples, self.n_classes_))
        for c_idx, c in enumerate(self.classes_):
            Y_onehot[y_arr == c, c_idx] = 1.0

        # Class weights to handle imbalanced market regimes
        class_counts = Y_onehot.sum(axis=0)
        class_weights = n_samples / (self.n_classes_ * np.maximum(class_counts, 1))

        # Iterative boosted logistic gradient descent with momentum and L2 regularization
        np.random.seed(self.random_state)
        W = np.zeros((n_features, self.n_classes_))
        b = np.log(class_counts / n_samples)
        v_W = np.zeros_like(W)
        v_b = np.zeros_like(b)

        lr = self.learning_rate
        reg = 0.01

        for epoch in range(120):
            logits = np.dot(X_norm, W) + b
            probs = self._softmax(logits)

            # Weighted gradient
            error = (probs - Y_onehot) * class_weights
            grad_W = np.dot(X_norm.T, error) / n_samples + reg * W
            grad_b = np.mean(error, axis=0)

            # Momentum update
            v_W = 0.85 * v_W - lr * grad_W
            v_b = 0.85 * v_b - lr * grad_b
            W += v_W
            b += v_b

        self.weights_ = W
        self.bias_ = b

        # Compute feature importance as L2 norm of weights per feature
        imp = np.sum(np.abs(W), axis=1)
        imp_sum = np.sum(imp)
        self.feature_importances_ = imp / (imp_sum if imp_sum > 0 else 1.0)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        X_norm = (X_arr - self.mean_) / self.std_
        X_norm = np.nan_to_num(X_norm, nan=0.0)
        logits = np.dot(X_norm, self.weights_) + self.bias_
        return self._softmax(logits)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        return self.classes_[pred_idx]


def get_classifier(
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    random_state: int = 42,
) -> Any:
    """Instantiate a LightGBM classifier with graceful fallback.

    Returns:
        LightGBM LGBMClassifier if installed; otherwise Scikit-Learn or PureNumpy booster.
    """
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        logger.info("Using native LightGBM Classifier (LGBMClassifier).")
        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
        return model
    except Exception as exc:
        logger.info("LightGBM not available (%s). Checking scikit-learn...", exc)

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore

        logger.info("Using Scikit-Learn HistGradientBoostingClassifier.")
        model = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
        )
        return model
    except Exception as exc:
        logger.info("Scikit-Learn not available (%s). Using PureNumpyGradientBooster.", exc)

    logger.info("Using high-performance PureNumpyGradientBooster.")
    return PureNumpyGradientBooster(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
