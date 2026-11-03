"""Model training and evaluation package for gadget-price-predictor."""

from src.model.classifier import get_classifier
from src.model.evaluate import (
    compute_classification_metrics,
    compute_feature_importances,
    evaluate_persistence_baseline,
    print_evaluation_summary,
)
from src.model.labeler import (
    CLASS_NAMES,
    LABEL_MAP,
    create_14day_target,
    prepare_features_and_target,
)
from src.model.train import predict_latest_forecast, train_and_evaluate
from src.model.walk_forward import generate_time_series_splits, run_walk_forward_validation

__all__ = [
    "get_classifier",
    "compute_classification_metrics",
    "compute_feature_importances",
    "evaluate_persistence_baseline",
    "print_evaluation_summary",
    "CLASS_NAMES",
    "LABEL_MAP",
    "create_14day_target",
    "prepare_features_and_target",
    "train_and_evaluate",
    "predict_latest_forecast",
    "generate_time_series_splits",
    "run_walk_forward_validation",
]
