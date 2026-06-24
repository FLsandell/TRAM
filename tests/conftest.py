from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def tiny_data(tmp_path: Path) -> dict[str, Path]:
    samples = [f"sample{i:02d}" for i in range(12)]
    matrix = pd.DataFrame({
        "ACC": ["scaffold_1_100", "scaffold_1_300", "scaffold_1_700", "scaffold_2_200"],
        **{
            sample: [int(i >= 6), i % 3, (i // 2) % 3, (i + 1) % 2]
            for i, sample in enumerate(samples)
        },
    })
    groups = pd.DataFrame({"ID": samples, "SP_CODE": ["Red"] * 6 + ["Fodder"] * 6})
    chromosomes = pd.DataFrame({
        "seqid": ["scaffold_1", "scaffold_2"],
        "end": [1000, 600],
        "CHR": ["CHR1", "CHR2"],
    })
    functions = pd.DataFrame({
        "#query": ["gene1.t1", "gene2.t1"],
        "Description": ["example one", "example two"],
        "GOs": ["GO:0000001", "GO:0000002"],
        "KEGG_ko": ["ko:K00001", "ko:K00002"],
        "PFAMs": ["PF00001", "PF00002"],
    })
    go = pd.DataFrame({
        "id": ["GO:0000001", "GO:0000002"],
        "name": ["term one", "term two"],
        "namespace": ["biological_process", "molecular_function"],
    })
    files = {
        "matrix": tmp_path / "matrix.tsv",
        "groups": tmp_path / "groups.tsv",
        "chromosome": tmp_path / "chromosomes.tsv",
        "gff": tmp_path / "genes.gff3",
        "database": tmp_path / "go.tsv",
        "function": tmp_path / "functions.tsv",
        "model": tmp_path / "model.json",
    }
    matrix.to_csv(files["matrix"], sep="\t", index=False)
    groups.to_csv(files["groups"], sep="\t", index=False)
    chromosomes.to_csv(files["chromosome"], sep="\t", index=False)
    functions.to_csv(files["function"], sep="\t", index=False)
    go.to_csv(files["database"], sep="\t", index=False)
    files["gff"].write_text(
        "##gff-version 3\n"
        "scaffold_1\ttest\tgene\t50\t180\t.\t+\t.\tID=gene1\n"
        "scaffold_2\ttest\tgene\t150\t250\t.\t-\t.\tID=gene2\n",
        encoding="utf-8",
    )
    files["model"].write_text(json.dumps({
        "n_estimators": 12,
        "max_depth": 4,
        "max_features": "sqrt",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
    }), encoding="utf-8")
    return files

