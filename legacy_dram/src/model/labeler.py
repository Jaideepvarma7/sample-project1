"""Module for creating the 14-day forward price change prediction target.

Target labels:
- UP   : Forward 14-day return > +3% (+0.03) -> class 2
- FLAT : Forward 14-day return between -3% and +3% -> class 1
- DOWN : Forward 14-day return < -3% (-0.03) -> class 0
"""

from __future__ import annotations

import logging
from typing import Final, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

THRESHOLD: Final[float] = 0.03  # 3% threshold
HORIZON_DAYS: Final[int] = 14

LABEL_MAP: Final[dict[int, str]] = {
    0: "DOWN",
    1: "FLAT",
    2: "UP",
}
REV_LABEL_MAP: Final[dict[str, int]] = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES: Final[List[str]] = ["DOWN", "FLAT", "UP"]


def create_14day_target(
    df: pd.DataFrame,
    price_col: str = "price",
    horizon: int = HORIZON_DAYS,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """Compute forward 14-day percentage return and discrete classification label.

    Forward return at index t:
        R_{t, t+14} = (price_{t+14} - price_t) / price_t

    Classification:
        - UP   : R > +0.03 (class 2)
        - DOWN : R < -0.03 (class 0)
        - FLAT : -0.03 <= R <= +0.03 (class 1)

    Note: The last `horizon` rows will have NaN for target_return and target_label,
    representing current unobserved future days suitable for live inference.

    Args:
        df: DataFrame containing date and price columns.
        price_col: Name of the RAM price column.
        horizon: Forecast window in days (default: 14).
        threshold: Percentage threshold (default: 0.03 for 3%).

    Returns:
        DataFrame with added 'target_return', 'target_label', and 'target_name'.
    """
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    future_price = out[price_col].shift(-horizon)
    current_price = out[price_col]

    # Calculate forward 14-day return
    fwd_return = (future_price - current_price) / current_price
    out["target_return"] = fwd_return.round(5)

    # Assign labels
    labels = pd.Series(np.nan, index=out.index)
    valid_mask = fwd_return.notna()

    up_mask = valid_mask & (fwd_return > threshold)
    down_mask = valid_mask & (fwd_return < -threshold)
    flat_mask = valid_mask & (~up_mask) & (~down_mask)

    labels[up_mask] = 2  # UP
    labels[flat_mask] = 1  # FLAT
    labels[down_mask] = 0  # DOWN

    out["target_label"] = labels
    out["target_name"] = labels.map(LABEL_MAP)

    logger.info(
        "Created 14-day targets: %d labeled rows, %d unobserved future rows. "
        "Label distribution:\n%s",
        valid_mask.sum(),
        (~valid_mask).sum(),
        out["target_name"].value_counts().to_dict(),
    )
    return out


def prepare_features_and_target(
    df: pd.DataFrame,
    exclude_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split dataset into feature matrix X, target vector y, and metadata.

    Args:
        df: Labeled dataset.
        exclude_cols: Columns to exclude from feature matrix.

    Returns:
        (X, y, df_clean) containing only rows with non-NaN labels.
    """
    default_exclude = [
        "date",
        "target_return",
        "target_label",
        "target_name",
        "price",  # Target series at t, use lags and rolling returns instead
    ]
    if exclude_cols:
        default_exclude.extend(exclude_cols)

    labeled = df.dropna(subset=["target_label"]).copy()
    feature_cols = [c for c in labeled.columns if c not in default_exclude]

    X = labeled[feature_cols].copy()
    y = labeled["target_label"].astype(int)

    return X, y, labeled
