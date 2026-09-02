"""Random-forest hyperparameter tuning stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from .io import load_dataset
from .selection import (
    DEFAULT_LOG_LOSS_TOLERANCE,
    DEFAULT_ROC_AUC_TOLERANCE,
    rf_complexity_key,
    select_best_candidate,
)


def tune_model(
    matrix: str | Path,
    groups: str | Path,
    target: str,
    group1: str,
    group2: str,
    output_prefix: str | Path,
    rounds: int,
    seed: int = 37,
    cv_folds: int = 5,
    jobs: int = -1,
    roc_auc_tolerance: float = DEFAULT_ROC_AUC_TOLERANCE,
    log_loss_tolerance: float = DEFAULT_LOG_LOSS_TOLERANCE,
) -> dict[str, object]:
    """Tune a random forest and write its parameters and best loss as JSON."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if roc_auc_tolerance < 0 or log_loss_tolerance < 0:
        raise ValueError("Selection tolerances must be non-negative")
    features, labels = load_dataset(matrix, groups, target, group1, group2)
    x_train, _, y_train, _ = train_test_split(
        features, labels, stratify=labels, random_state=seed
    )
    _, training_counts = np.unique(y_train, return_counts=True)
    if training_counts.min() < cv_folds:
        raise ValueError(
            f"Each group needs at least {cv_folds} training samples for {cv_folds}-fold CV"
        )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    encoded_train = OneHotEncoder(handle_unknown="ignore").fit_transform(x_train)
    space = {
        "n_estimators": hp.quniform("n_estimators", 200, 999, 1),
        "max_depth": hp.quniform("max_depth", 1, 64, 1),
        "max_features": hp.choice("max_features", ["sqrt"]),
        "min_samples_split": hp.quniform("min_samples_split", 5, 31, 1),
        "min_samples_leaf": hp.quniform("min_samples_leaf", 5, 31, 1),
        "bootstrap": hp.choice("bootstrap", [True]),
    }

    def objective(candidate: dict[str, object]) -> dict[str, object]:
        params = _coerce_parameters(candidate)
        mean_auc, oof_log_loss = _cross_validated_metrics(
            encoded_train,
            y_train,
            cv,
            params,
            seed=seed,
            jobs=jobs,
        )
        return {
            "loss": -mean_auc,
            "status": STATUS_OK,
            "mean_cv_roc_auc": mean_auc,
            "oof_log_loss": oof_log_loss,
            "parameters": params,
        }

    trials = Trials()
    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=rounds,
        trials=trials,
        rstate=np.random.default_rng(seed),
        return_argmin=False,
    )
    candidates = [
        {
            "trial": int(trial["tid"]),
            "mean_cv_roc_auc": float(trial["result"]["mean_cv_roc_auc"]),
            "oof_log_loss": float(trial["result"]["oof_log_loss"]),
            "parameters": trial["result"]["parameters"],
            "complexity_key": list(rf_complexity_key(trial["result"]["parameters"])),
        }
        for trial in trials.trials
    ]
    selected = select_best_candidate(
        candidates,
        roc_auc_tolerance=roc_auc_tolerance,
        log_loss_tolerance=log_loss_tolerance,
    )
    best = dict(selected["parameters"])

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.model").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    Path(f"{prefix}.loss").write_text(
        f"{-float(selected['mean_cv_roc_auc'])}\n",
        encoding="utf-8",
    )
    audit = {
        "selection_policy": {
            "primary": "highest mean CV ROC-AUC",
            "secondary": "lowest out-of-fold log loss among near-best ROC-AUC trials",
            "tertiary": "most regularized random forest when both metrics are effectively tied",
            "roc_auc_tolerance": roc_auc_tolerance,
            "log_loss_tolerance": log_loss_tolerance,
            "complexity_order": [
                "larger min_samples_leaf",
                "smaller max_depth",
                "fewer max_features",
                "larger min_samples_split",
                "fewer n_estimators",
            ],
        },
        "selected_trial": selected,
        "trials": candidates,
    }
    Path(f"{prefix}.tuning.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    return best


def _cross_validated_metrics(
    features: Any,
    labels: Any,
    cv: StratifiedKFold,
    parameters: dict[str, object],
    *,
    seed: int,
    jobs: int,
) -> tuple[float, float]:
    """Return mean fold ROC-AUC and log loss from pooled OOF probabilities."""
    label_values = np.asarray(labels)
    classes = np.unique(label_values)
    if len(classes) != 2:
        raise ValueError("ROC-AUC tuning requires exactly two phenotype groups")

    oof_probabilities = np.full((len(label_values), len(classes)), np.nan, dtype=float)
    auc_scores: list[float] = []
    class_columns = {label: index for index, label in enumerate(classes)}

    for training_indices, validation_indices in cv.split(features, label_values):
        classifier = RandomForestClassifier(
            **parameters,
            random_state=seed,
            n_jobs=jobs,
        )
        classifier.fit(features[training_indices], label_values[training_indices])
        fold_probabilities = classifier.predict_proba(features[validation_indices])
        for model_column, label in enumerate(classifier.classes_):
            oof_probabilities[validation_indices, class_columns[label]] = fold_probabilities[:, model_column]

        positive = (label_values[validation_indices] == classes[1]).astype(int)
        auc_scores.append(
            float(roc_auc_score(positive, oof_probabilities[validation_indices, 1]))
        )

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("Cross-validation did not produce an OOF probability for every sample")
    return (
        float(np.mean(auc_scores)),
        float(log_loss(label_values, oof_probabilities, labels=classes)),
    )


def _coerce_parameters(parameters: dict[str, object]) -> dict[str, object]:
    result = dict(parameters)
    for key in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"):
        result[key] = int(result[key])
    result["bootstrap"] = bool(result["bootstrap"])
    return result
