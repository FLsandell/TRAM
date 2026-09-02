"""Command-line interface for TRAM."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .selection import DEFAULT_LOG_LOSS_TOLERANCE, DEFAULT_ROC_AUC_TOLERANCE


def _dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--matrix", type=Path, required=True, help="Transposed tab-separated SNP matrix")
    parser.add_argument("-g", "--groups", type=Path, required=True, help="Tab-separated sample phenotype table")
    parser.add_argument("-t", "--target", default="SP_CODE", help="Phenotype column (default: SP_CODE)")
    parser.add_argument("-1", "--group1", required=True, help="First phenotype value")
    parser.add_argument("-2", "--group2", required=True, help="Second phenotype value")


def _compute_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=37, help="Random seed (default: 37)")
    parser.add_argument("--jobs", type=int, default=-1, help="Random-forest parallel jobs (default: all CPUs)")


def _sliding_arguments(parser: argparse.ArgumentParser, include_summary: bool = True) -> None:
    if include_summary:
        parser.add_argument("-s", "--summary", type=Path, required=True, help="RF importance summary")
    parser.add_argument("-c", "--chromosome", type=Path, required=True, help="Scaffold/chromosome length table")
    parser.add_argument("--gff", type=Path, required=True, help="GFF3 gene annotation")
    parser.add_argument("-d", "--database", type=Path, required=True, help="GO term database")
    parser.add_argument("-f", "--function", type=Path, required=True, help="Functional annotation table")
    parser.add_argument("-r", "--repeat-fraction", type=float, required=True, help="Repetitive genome fraction in [0, 1)")
    parser.add_argument("--randomizations", type=int, default=999, help="Null randomizations (default: 999)")
    parser.add_argument("--workers", type=int, default=1, help="Randomization worker processes (default: 1)")
    parser.add_argument("--window-size", type=int, default=10_000, help="Window size in bp (default: 10000)")
    parser.add_argument("--step-size", type=int, default=5_000, help="Window step in bp (default: 5000)")
    parser.add_argument("--significance-quantile", type=float, default=0.999, help="Null quantile (default: 0.999)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tram", description="Trait-associated region analysis with random forests")
    parser.add_argument("--version", action="version", version=f"TRAM {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run tuning, repeated modeling, and sliding windows")
    _dataset_arguments(run)
    _compute_arguments(run)
    _sliding_arguments(run, include_summary=False)
    run.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    run.add_argument("--rounds", type=int, required=True, help="Hyperparameter evaluations")
    run.add_argument(
        "--roc-auc-tolerance",
        type=float,
        default=DEFAULT_ROC_AUC_TOLERANCE,
        help="Near-best ROC-AUC tolerance for log-loss tie-breaking (default: 0.001)",
    )
    run.add_argument(
        "--log-loss-tolerance",
        type=float,
        default=DEFAULT_LOG_LOSS_TOLERANCE,
        help="Effective log-loss tie tolerance for RF-complexity tie-breaking (default: 0.001)",
    )
    run.add_argument("--cv-folds", type=int, default=5, help="Tuning CV folds (default: 5)")
    run.add_argument("--replicates", type=int, default=100, help="Repeated models (default: 100)")
    run.add_argument("--test-size", type=float, default=0.25, help="Test fraction (default: 0.25)")

    tune = commands.add_parser("tune", help="Tune random-forest hyperparameters")
    _dataset_arguments(tune)
    _compute_arguments(tune)
    tune.add_argument("-o", "--output-prefix", type=Path, required=True, help="Model output prefix")
    tune.add_argument("--rounds", type=int, required=True, help="Hyperparameter evaluations")
    tune.add_argument(
        "--roc-auc-tolerance",
        type=float,
        default=DEFAULT_ROC_AUC_TOLERANCE,
        help="Near-best ROC-AUC tolerance for log-loss tie-breaking (default: 0.001)",
    )
    tune.add_argument(
        "--log-loss-tolerance",
        type=float,
        default=DEFAULT_LOG_LOSS_TOLERANCE,
        help="Effective log-loss tie tolerance for RF-complexity tie-breaking (default: 0.001)",
    )
    tune.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds (default: 5)")

    model = commands.add_parser("model", help="Train repeated random-forest models")
    _dataset_arguments(model)
    _compute_arguments(model)
    model.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    model.add_argument("-p", "--model", type=Path, required=True, help="Tuned model JSON file")
    model.add_argument("--replicates", type=int, default=100, help="Repeated models (default: 100)")
    model.add_argument("--test-size", type=float, default=0.25, help="Test fraction (default: 0.25)")

    sliding = commands.add_parser("sliding-window", help="Run sliding-window analysis and annotation")
    _sliding_arguments(sliding)
    sliding.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    sliding.add_argument("--seed", type=int, default=37, help="Random seed (default: 37)")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = vars(build_parser().parse_args(argv))
    command = arguments.pop("command")
    if command == "run":
        from .pipeline import run_pipeline

        outputs = run_pipeline(**arguments)
        for name, path in outputs.items():
            print(f"{name}: {path}")
    elif command == "tune":
        from .tuning import tune_model

        tune_model(**arguments)
    elif command == "model":
        from .modeling import train_replicates

        summary = train_replicates(**arguments)
        print(f"summary: {summary}")
    elif command == "sliding-window":
        from .sliding_window import sliding_window_analysis

        threshold = sliding_window_analysis(**arguments)
        print(f"significance threshold: {threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
