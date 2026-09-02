"""Deterministic hierarchical selection for tuned random-forest models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import inf
from typing import Any

DEFAULT_ROC_AUC_TOLERANCE = 0.001
DEFAULT_LOG_LOSS_TOLERANCE = 0.001


def rf_complexity_key(parameters: Mapping[str, Any]) -> tuple[float, ...]:
    """Return a key where smaller values represent a more regularized RF.

    Minimum leaf size is deliberately considered before depth and features.
    Tree count is last because it primarily changes compute cost and variance,
    rather than model flexibility.
    """
    minimum_leaf = float(parameters.get("min_samples_leaf", 1))
    maximum_depth = parameters.get("max_depth")
    depth = inf if maximum_depth is None else float(maximum_depth)
    features = _max_features_key(parameters.get("max_features"))
    minimum_split = float(parameters.get("min_samples_split", 2))
    trees = float(parameters.get("n_estimators", 100))
    return (-minimum_leaf, depth, features, -minimum_split, trees)


def select_best_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    roc_auc_tolerance: float = DEFAULT_ROC_AUC_TOLERANCE,
    log_loss_tolerance: float = DEFAULT_LOG_LOSS_TOLERANCE,
) -> Mapping[str, Any]:
    """Select by ROC-AUC, then OOF log loss, then RF regularization."""
    if roc_auc_tolerance < 0 or log_loss_tolerance < 0:
        raise ValueError("Selection tolerances must be non-negative")
    if not candidates:
        raise ValueError("At least one tuning candidate is required")

    best_auc = max(float(candidate["mean_cv_roc_auc"]) for candidate in candidates)
    auc_pool = [
        candidate
        for candidate in candidates
        if float(candidate["mean_cv_roc_auc"]) >= best_auc - roc_auc_tolerance
    ]

    best_log_loss = min(float(candidate["oof_log_loss"]) for candidate in auc_pool)
    metric_tie_pool = [
        candidate
        for candidate in auc_pool
        if float(candidate["oof_log_loss"]) <= best_log_loss + log_loss_tolerance
    ]

    return min(
        metric_tie_pool,
        key=lambda candidate: (
            rf_complexity_key(candidate["parameters"]),
            -float(candidate["mean_cv_roc_auc"]),
            float(candidate["oof_log_loss"]),
            int(candidate.get("trial", 0)),
        ),
    )


def _max_features_key(value: Any) -> float:
    if value == "log2":
        return 0.0
    if value in {"sqrt", "auto"}:
        return 1.0
    if value is None:
        return 4.0
    if isinstance(value, float):
        return 2.0 + value
    if isinstance(value, int):
        return 3.0 + value
    return 5.0
