"""Random-forest hyperparameter tuning stage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder

from .io import load_dataset


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
) -> dict[str, object]:
    """Tune a random forest and write its parameters and best loss as JSON."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
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
        classifier = RandomForestClassifier(**params, random_state=seed, n_jobs=jobs)
        score = cross_val_score(classifier, encoded_train, y_train, scoring="roc_auc", cv=cv).mean()
        return {"loss": -float(score), "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=rounds,
        trials=trials,
        rstate=np.random.default_rng(seed),
        return_argmin=False,
    )
    best = _coerce_parameters(best)
    best_loss = min(float(trial["result"]["loss"]) for trial in trials.trials)

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.model").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    Path(f"{prefix}.loss").write_text(f"{best_loss}\n", encoding="utf-8")
    return best


def _coerce_parameters(parameters: dict[str, object]) -> dict[str, object]:
    result = dict(parameters)
    for key in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"):
        result[key] = int(result[key])
    result["bootstrap"] = bool(result["bootstrap"])
    return result
