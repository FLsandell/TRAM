"""End-to-end TRAM pipeline orchestration."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .io import group_name
from .modeling import train_replicates
from .selection import DEFAULT_LOG_LOSS_TOLERANCE, DEFAULT_ROC_AUC_TOLERANCE
from .sliding_window import sliding_window_analysis
from .tuning import tune_model


def run_pipeline(**options: object) -> dict[str, str]:
    """Run all three stages and return the principal output paths."""
    options = dict(options)
    options.setdefault("roc_auc_tolerance", DEFAULT_ROC_AUC_TOLERANCE)
    options.setdefault("log_loss_tolerance", DEFAULT_LOG_LOSS_TOLERANCE)
    output = Path(str(options["output"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_prefix = output / "rf_tuned_model"
    common = {key: options[key] for key in ("matrix", "groups", "target", "group1", "group2")}
    tune_model(
        **common,
        output_prefix=model_prefix,
        rounds=int(options["rounds"]),
        seed=int(options["seed"]),
        cv_folds=int(options["cv_folds"]),
        jobs=int(options["jobs"]),
        roc_auc_tolerance=float(options["roc_auc_tolerance"]),
        log_loss_tolerance=float(options["log_loss_tolerance"]),
    )
    summary = train_replicates(
        **common,
        output=output,
        model=f"{model_prefix}.model",
        replicates=int(options["replicates"]),
        test_size=float(options["test_size"]),
        seed=int(options["seed"]),
        jobs=int(options["jobs"]),
    )
    sliding_output = output / f"sliding_window_{group_name(str(options['group1']), str(options['group2']))}"
    print("TRAM: starting sliding-window analysis", flush=True)
    sliding_window_analysis(
        summary=summary,
        chromosome=options["chromosome"],
        gff=options["gff"],
        database=options["database"],
        function=options["function"],
        output=sliding_output,
        repeat_fraction=float(options["repeat_fraction"]),
        randomizations=int(options["randomizations"]),
        workers=int(options["workers"]),
        window_size=int(options["window_size"]),
        step_size=int(options["step_size"]),
        significance_quantile=float(options["significance_quantile"]),
        seed=int(options["seed"]),
    )
    result = {
        "model": str(Path(f"{model_prefix}.model")),
        "tuning": str(Path(f"{model_prefix}.tuning.json")),
        "summary": str(summary),
        "sliding_output": str(sliding_output),
    }
    metadata = {
        "tram_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in options.items()},
        "outputs": result,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return result
