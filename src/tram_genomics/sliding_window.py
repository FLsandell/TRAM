"""Sliding-window significance analysis and functional annotation."""

from __future__ import annotations

import collections
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "tram_matplotlib")
import matplotlib.pyplot as plt

from .io import require_columns


def _window_sums(positions: np.ndarray, weights: np.ndarray, starts: np.ndarray, window_size: int) -> np.ndarray:
    """Sum weighted positions into inclusive, overlapping windows."""
    changes = np.zeros(len(starts) + 1, dtype=float)
    left = np.searchsorted(starts, positions - window_size, side="left")
    right = np.searchsorted(starts, positions, side="right")
    valid = right > left
    np.add.at(changes, left[valid], weights[valid])
    np.add.at(changes, right[valid], -weights[valid])
    return np.cumsum(changes[:-1])


def _randomized_windows(task: tuple[int, np.ndarray, int, np.ndarray, int]) -> np.ndarray:
    seed, weights, genome_length, starts, window_size = task
    rng = np.random.default_rng(seed)
    positions = rng.integers(1, genome_length + 1, size=len(weights))
    return _window_sums(positions, weights, starts, window_size)


def _read_gff(path: str | Path) -> pd.DataFrame:
    columns = ["seqid", "source", "TYPE", "START", "STOP", "score", "strand", "phase", "attributes"]
    return pd.read_csv(path, sep="\t", comment="#", header=None, names=columns)


def _read_function_annotations(path: str | Path) -> pd.DataFrame:
    """Read eggNOG annotation tables with or without leading metadata lines."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.startswith("#query\t"):
                return pd.read_csv(path, sep="\t", skiprows=index)
    return pd.read_csv(path, sep="\t")


def sliding_window_analysis(
    summary: str | Path,
    chromosome: str | Path,
    gff: str | Path,
    database: str | Path,
    function: str | Path,
    output: str | Path,
    repeat_fraction: float,
    randomizations: int = 999,
    workers: int = 1,
    window_size: int = 10_000,
    step_size: int = 5_000,
    significance_quantile: float = 0.999,
    seed: int = 20_747,
) -> float:
    """Run genomic windows, annotate significant genes, and return the threshold."""
    if not 0 <= repeat_fraction < 1:
        raise ValueError("repeat_fraction must be in the interval [0, 1)")
    if randomizations < 1 or workers < 1 or window_size < 1 or step_size < 1:
        raise ValueError("randomizations, workers, window_size, and step_size must be positive")
    if not 0 < significance_quantile < 1:
        raise ValueError("significance_quantile must be between 0 and 1")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    variants = pd.read_csv(summary, sep="\t")
    chromosomes = pd.read_csv(chromosome, sep="\t")
    functions = _read_function_annotations(function)
    go_database = pd.read_csv(database, sep="\t")
    require_columns(variants, {"seqid", "POS", "VarImp_sum"}, summary)
    require_columns(chromosomes, {"seqid", "end", "CHR"}, chromosome)
    require_columns(functions, {"#query", "Description", "GOs", "KEGG_ko", "PFAMs"}, function)
    require_columns(go_database, {"id"}, database)

    effective_length = int(chromosomes["end"].sum() * (1 - repeat_fraction))
    if effective_length < 1:
        raise ValueError("The non-repetitive genome length must be positive")
    random_starts = np.arange(1, effective_length + 1, step_size, dtype=np.int64)
    weights = variants["VarImp_sum"].to_numpy(dtype=float)
    mmap_path = output / ".randomized_windows.tmp"
    simulations = np.memmap(mmap_path, dtype="float64", mode="w+", shape=(randomizations, len(random_starts)))
    tasks = ((seed + i, weights, effective_length, random_starts, window_size) for i in range(randomizations))
    try:
        if workers == 1:
            results = map(_randomized_windows, tasks)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            results = executor.map(_randomized_windows, tasks)
        try:
            for index, values in enumerate(results):
                simulations[index] = values
        finally:
            if workers != 1:
                executor.shutdown()
        simulations.flush()
        threshold = float(np.quantile(simulations, significance_quantile, axis=0).mean())
    finally:
        del simulations
        if mmap_path.exists():
            mmap_path.unlink()

    windows: list[pd.DataFrame] = []
    chromosome_offsets: dict[str, int] = collections.defaultdict(int)
    for row in chromosomes.itertuples(index=False):
        starts = np.arange(1, int(row.end) + 1, step_size, dtype=np.int64)
        scaffold = variants.loc[variants["seqid"] == row.seqid]
        sums = _window_sums(
            scaffold["POS"].to_numpy(dtype=np.int64),
            scaffold["VarImp_sum"].to_numpy(dtype=float),
            starts,
            window_size,
        )
        frame = pd.DataFrame({"CHR": row.CHR, "seqid": row.seqid, "POS": starts, "VarImp_sum": sums})
        frame["POS_COMT"] = chromosome_offsets[row.CHR] + np.arange(len(frame)) * step_size
        chromosome_offsets[row.CHR] += len(frame) * step_size
        windows.append(frame)
    all_windows = pd.concat(windows, ignore_index=True)
    all_windows.to_csv(output / "windows.txt", sep="\t", index=False)
    _plot_windows(all_windows, threshold, output / "windows_plot.png")

    genes = _annotate_genes(all_windows, threshold, _read_gff(gff), functions, window_size)
    genes.to_csv(output / "genes.txt", sep="\t", index=False)
    interesting_columns = ["#query", "Description", "GOs", "KEGG_ko", "PFAMs"]
    interesting = genes.reindex(columns=interesting_columns)
    interesting.to_csv(output / "genes_interesting.txt", sep="\t", index=False)

    go_terms = [term for value in interesting["GOs"].dropna() for term in str(value).split(",") if term]
    counts = pd.DataFrame(collections.Counter(go_terms).items(), columns=["id", "COUNT"])
    translated = counts.merge(go_database, on="id", how="left").sort_values("COUNT", ascending=False)
    translated.to_csv(output / "translated_go_terms.txt", sep="\t", index=False)
    (output / "sig_value.txt").write_text(f"{threshold}\n", encoding="utf-8")
    return threshold


def _annotate_genes(
    windows: pd.DataFrame,
    threshold: float,
    gff: pd.DataFrame,
    functions: pd.DataFrame,
    window_size: int,
) -> pd.DataFrame:
    genes = gff.loc[gff["TYPE"] == "gene"].copy()
    identifiers: set[str] = set()
    for window in windows.loc[windows["VarImp_sum"] > threshold].itertuples(index=False):
        overlaps = genes.loc[
            (genes["seqid"] == window.seqid)
            & (genes["START"] <= window.POS + window_size)
            & (genes["STOP"] >= window.POS)
        ]
        for attributes in overlaps["attributes"].dropna():
            fields = dict(part.split("=", 1) for part in str(attributes).split(";") if "=" in part)
            if "ID" in fields:
                identifiers.add(fields["ID"])
    if not identifiers:
        return functions.iloc[0:0].copy()
    query = functions["#query"].astype(str)
    mask = pd.Series(False, index=functions.index)
    for identifier in identifiers:
        mask |= query.eq(identifier) | query.str.startswith(f"{identifier}.")
    return functions.loc[mask].drop_duplicates().copy()


def _plot_windows(windows: pd.DataFrame, threshold: float, destination: Path) -> None:
    names = list(windows["CHR"].drop_duplicates())
    columns = 2
    rows = max(1, (len(names) + columns - 1) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows), constrained_layout=True, squeeze=False)
    maximum = max(float(windows["VarImp_sum"].max()) * 1.1, threshold * 1.1, 1e-12)
    for axis, name in zip(axes.flat, names):
        frame = windows.loc[windows["CHR"] == name]
        axis.plot(frame["POS_COMT"], frame["VarImp_sum"], color="#285f8f", linewidth=1)
        axis.axhline(threshold, color="#b43c35", linestyle="--", linewidth=1)
        axis.set(title=name, xlabel="Position (bp)", ylabel="Summed importance", ylim=(0, maximum))
    for axis in list(axes.flat)[len(names):]:
        axis.axis("off")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
