import numpy as np
from sklearn.model_selection import StratifiedKFold

from tram_genomics.selection import rf_complexity_key, select_best_candidate
from tram_genomics.tuning import _cross_validated_metrics


def _candidate(trial, roc_auc, log_loss, **parameter_overrides):
    parameters = {
        "n_estimators": 500,
        "max_depth": 12,
        "max_features": "sqrt",
        "min_samples_split": 10,
        "min_samples_leaf": 10,
        "bootstrap": True,
    }
    parameters.update(parameter_overrides)
    return {
        "trial": trial,
        "mean_cv_roc_auc": roc_auc,
        "oof_log_loss": log_loss,
        "parameters": parameters,
    }


def test_roc_auc_is_primary_outside_tolerance():
    best_auc = _candidate(0, 0.9000, 0.50, min_samples_leaf=20, max_depth=5)
    better_log_loss = _candidate(1, 0.8988, 0.10, min_samples_leaf=25, max_depth=4)

    selected = select_best_candidate(
        [best_auc, better_log_loss],
        roc_auc_tolerance=0.001,
        log_loss_tolerance=0.001,
    )

    assert selected["trial"] == 0


def test_oof_log_loss_breaks_near_best_roc_auc_tie():
    best_auc = _candidate(0, 0.9000, 0.50, min_samples_leaf=20, max_depth=5)
    better_log_loss = _candidate(1, 0.8995, 0.40, min_samples_leaf=5, max_depth=30)

    selected = select_best_candidate(
        [best_auc, better_log_loss],
        roc_auc_tolerance=0.001,
        log_loss_tolerance=0.001,
    )

    assert selected["trial"] == 1


def test_regularization_breaks_effective_metric_ties():
    simpler = _candidate(0, 0.9000, 0.4000, min_samples_leaf=20, max_depth=5)
    marginally_better_log_loss = _candidate(
        1,
        0.8995,
        0.3995,
        min_samples_leaf=5,
        max_depth=30,
    )

    selected = select_best_candidate(
        [simpler, marginally_better_log_loss],
        roc_auc_tolerance=0.001,
        log_loss_tolerance=0.001,
    )

    assert selected["trial"] == 0


def test_rf_complexity_prefers_leaf_depth_features_and_split_before_trees():
    baseline = _candidate(0, 0.9, 0.4)["parameters"]

    assert rf_complexity_key({**baseline, "min_samples_leaf": 20}) < rf_complexity_key(baseline)
    assert rf_complexity_key({**baseline, "max_depth": 5}) < rf_complexity_key(baseline)
    assert rf_complexity_key({**baseline, "max_features": "log2"}) < rf_complexity_key(baseline)
    assert rf_complexity_key({**baseline, "min_samples_split": 20}) < rf_complexity_key(baseline)
    assert rf_complexity_key({**baseline, "n_estimators": 400}) < rf_complexity_key(baseline)


def test_selector_rejects_negative_tolerances():
    candidate = _candidate(0, 0.9, 0.4)

    try:
        select_best_candidate([candidate], roc_auc_tolerance=-0.001, log_loss_tolerance=0.001)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative tolerances must be rejected")


def test_cross_validated_metrics_include_true_oof_log_loss():
    features = np.array([[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1]])
    labels = np.array(["A"] * 6 + ["B"] * 6)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    parameters = {
        "n_estimators": 20,
        "max_depth": 3,
        "max_features": "sqrt",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
    }

    mean_auc, oof_log_loss = _cross_validated_metrics(
        features,
        labels,
        cv,
        parameters,
        seed=7,
        jobs=1,
    )

    assert mean_auc > 0.95
    assert 0 <= oof_log_loss < 0.2
