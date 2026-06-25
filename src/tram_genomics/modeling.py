"""Repeated random-forest modeling and feature-importance summaries."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .io import group_name, load_dataset


def load_model_parameters(path: str | Path) -> dict[str, object]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        parameters = json.loads(text)
    except json.JSONDecodeError:
        parameters = ast.literal_eval(text)
    if not isinstance(parameters, dict):
        raise ValueError("Model parameter file must contain an object/dictionary")
    for key in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"):
        if key in parameters and parameters[key] is not None:
            parameters[key] = int(parameters[key])
    return parameters


def train_replicates(
    matrix: str | Path,
    groups: str | Path,
    target: str,
    group1: str,
    group2: str,
    output: str | Path,
    model: str | Path,
    replicates: int = 100,
    test_size: float = 0.25,
    seed: int = 37,
    jobs: int = -1,
) -> Path:
    """Train repeated models and return the feature-importance summary path."""
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    features, labels = load_dataset(matrix, groups, target, group1, group2)
    parameters = load_model_parameters(model)
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    comparison = group_name(group1, group2)
    outpath = Path(output) / comparison / "replicates"
    outpath.mkdir(parents=True, exist_ok=True)

    importances: list[pd.DataFrame] = []
    reports: list[pd.DataFrame] = []
    mistakes: list[pd.DataFrame] = []
    feature_names = list(features.columns)
    _print_progress(0, replicates)

    for run in range(1, replicates + 1):
        run_seed = seed + run - 1
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            encoded_labels,
            stratify=encoded_labels,
            test_size=test_size,
            random_state=run_seed,
        )
        one_hot = OneHotEncoder(handle_unknown="ignore")
        train_encoded = one_hot.fit_transform(x_train)
        test_encoded = one_hot.transform(x_test)
        classifier = RandomForestClassifier(
            **parameters, random_state=run_seed, n_jobs=jobs
        ).fit(train_encoded, y_train)
        predictions = classifier.predict(test_encoded)

        feature_importance = pd.DataFrame({
            "SNP": feature_names,
            "VarImp": _snp_importances(classifier, one_hot),
            "Run": run,
        })
        importances.append(feature_importance.loc[feature_importance["VarImp"] > 0])

        report = pd.DataFrame(classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        )).T
        report["Run"] = run
        reports.append(report)

        incorrect = x_test.loc[y_test != predictions].copy()
        incorrect[target] = encoder.inverse_transform(y_test[y_test != predictions])
        incorrect["Predicted_as"] = encoder.inverse_transform(predictions[y_test != predictions])
        incorrect["Run"] = run
        mistakes.append(incorrect)
        _print_progress(run, replicates)

    stem = comparison
    all_importances = pd.concat(importances, ignore_index=True)
    summary = _summarize_importances(all_importances)
    summary_path = outpath / f"RF_varImpSummary_{stem}.txt"
    summary.to_csv(summary_path, sep="\t", index=False)
    all_importances.to_csv(outpath / f"RF_VarImp_{stem}.csv", sep="\t", index=False)
    _write_model_reports(outpath, stem, all_importances, reports, mistakes)
    return summary_path


def _snp_importances(classifier: RandomForestClassifier, encoder: OneHotEncoder) -> np.ndarray:
    offsets = np.cumsum([0, *[len(categories) for categories in encoder.categories_]])
    return np.add.reduceat(classifier.feature_importances_, offsets[:-1])


def _summarize_importances(importances: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "SNP",
        "seqid",
        "POS",
        "VarImp_mean",
        "VarImp_max",
        "VarImp_min",
        "VarImp_std",
        "VarImp_sum",
        "VarImp_median",
        "VarImp_count",
    ]
    if importances.empty:
        return pd.DataFrame(columns=columns)

    summary = importances.groupby("SNP")["VarImp"].agg(
        ["mean", "max", "min", "std", "sum", "median", "count"]
    ).reset_index()
    summary.columns = [column for column in columns if column not in {"seqid", "POS"}]
    coordinates = summary["SNP"].str.rsplit("_", n=1, expand=True)
    if coordinates.shape[1] != 2 or not pd.to_numeric(coordinates[1], errors="coerce").notna().all():
        raise ValueError("Every SNP identifier must end in '_<integer position>'")
    summary.insert(1, "seqid", coordinates[0])
    summary.insert(2, "POS", coordinates[1].astype(int))
    return summary


def _write_model_reports(
    outpath: Path,
    stem: str,
    importances: pd.DataFrame,
    reports: list[pd.DataFrame],
    mistakes: list[pd.DataFrame],
) -> None:
    all_reports = pd.concat(reports)
    all_reports.to_csv(outpath / f"RF_classReport_{stem}.txt", sep="\t")

    all_mistakes = pd.concat(mistakes) if mistakes else pd.DataFrame()
    all_mistakes.to_csv(outpath / f"RF_misclass_{stem}.txt", sep="\t", index=True)

    occurrences = (
        all_mistakes.groupby(level=0).size().rename("Occurrence")
        if not all_mistakes.empty
        else pd.Series(dtype=int, name="Occurrence")
    )
    occurrences.to_csv(outpath / f"RF_misclassSummary_{stem}.txt", sep="\t")

    accuracy = all_reports.loc[all_reports.index == "accuracy", "f1-score"]
    accuracy.agg(["mean", "median", "std"]).to_csv(outpath / f"RF_accuracySummary_{stem}.txt", sep="\t")
    importances["VarImp"].agg(["median", "mean", "std", "max", "min", "sum", "count"]).to_csv(
        outpath / f"RF_genomeVarImp_Summary_{stem}.txt", sep="\t"
    )


def _print_progress(done: int, total: int) -> None:
    width = 30
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if done == total else "\r"
    print(f"TRAM models: [{bar}] {done}/{total}", end=end, flush=True)
