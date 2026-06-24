from pathlib import Path

from tram_genomics.modeling import train_replicates
from tram_genomics.pipeline import run_pipeline
from tram_genomics.sliding_window import sliding_window_analysis


def test_modeling_and_sliding_window_smoke(tiny_data, tmp_path):
    output = tmp_path / "output"
    summary = train_replicates(
        matrix=tiny_data["matrix"],
        groups=tiny_data["groups"],
        target="SP_CODE",
        group1="Red",
        group2="Fodder",
        output=output,
        model=tiny_data["model"],
        replicates=2,
        jobs=1,
    )
    assert summary.exists()

    sliding_output = output / "sliding"
    threshold = sliding_window_analysis(
        summary=summary,
        chromosome=tiny_data["chromosome"],
        gff=tiny_data["gff"],
        database=tiny_data["database"],
        function=tiny_data["function"],
        output=sliding_output,
        repeat_fraction=0.1,
        randomizations=5,
        workers=1,
        window_size=200,
        step_size=100,
        significance_quantile=0.9,
        seed=7,
    )
    assert threshold >= 0
    assert (sliding_output / "windows_plot.png").exists()
    assert (sliding_output / "translated_go_terms.txt").exists()


def test_complete_pipeline_writes_metadata(tiny_data, tmp_path):
    output = tmp_path / "complete"
    result = run_pipeline(
        matrix=tiny_data["matrix"],
        groups=tiny_data["groups"],
        target="SP_CODE",
        group1="Red",
        group2="Fodder",
        chromosome=tiny_data["chromosome"],
        gff=tiny_data["gff"],
        database=tiny_data["database"],
        function=tiny_data["function"],
        repeat_fraction=0.1,
        output=output,
        rounds=1,
        cv_folds=3,
        replicates=1,
        test_size=0.25,
        seed=7,
        jobs=1,
        randomizations=3,
        workers=1,
        window_size=200,
        step_size=100,
        significance_quantile=0.9,
    )
    assert all(Path(path).exists() for path in result.values())
    assert (output / "run_metadata.json").exists()
