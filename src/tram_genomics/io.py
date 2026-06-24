"""Input loading and validation shared by TRAM stages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def require_columns(frame: pd.DataFrame, columns: set[str], source: str | Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required column(s): {', '.join(missing)}")


def load_dataset(
    matrix: str | Path,
    groups: str | Path,
    target: str,
    group1: str,
    group2: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the transposed SNP matrix and align phenotypes by sample ID."""
    matrix = Path(matrix)
    groups = Path(groups)
    genotype = pd.read_csv(matrix, sep="\t")
    require_columns(genotype, {"ACC"}, matrix)

    phenotype = pd.read_csv(groups, sep="\t")
    require_columns(phenotype, {"ID", target}, groups)
    if phenotype["ID"].duplicated().any():
        raise ValueError(f"{groups} contains duplicate sample IDs")

    genotype = genotype.set_index("ACC").T
    genotype.index = genotype.index.astype(str)
    genotype = genotype.apply(pd.to_numeric, errors="raise")
    phenotype = phenotype.assign(ID=phenotype["ID"].astype(str)).set_index("ID")

    missing = genotype.index.difference(phenotype.index)
    if not missing.empty:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Phenotypes are missing for {len(missing)} sample(s): {preview}")

    labels = phenotype.loc[genotype.index, target]
    selected = labels.isin([group1, group2])
    genotype = genotype.loc[selected]
    labels = labels.loc[selected]

    counts = labels.value_counts()
    absent = [group for group in (group1, group2) if group not in counts]
    if absent:
        raise ValueError(f"No samples found for group(s): {', '.join(absent)}")
    if counts.min() < 2:
        raise ValueError("Each selected group must contain at least two samples")
    return genotype.astype("int8"), labels


def group_name(group1: str, group2: str) -> str:
    """Return a filesystem-friendly, stable comparison name."""
    clean = lambda value: "_".join(value.strip().split()).replace("/", "_")
    return f"{clean(group1)}-{clean(group2)}"

