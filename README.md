# Tree Based Association Mapping (TRAM)

TRAM identifies trait-associated genomic regions by tuning a random forest,
summarizing SNP importance across repeated models, and testing genomic sliding
windows against a randomized null distribution.

## Installation

TRAM requires Python 3.10 or newer.

```bash
git clone https://github.com/FLsandell/TRAM.git
cd TRAM
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For development and testing, use `python -m pip install -e ".[test]"`.

## Complete Pipeline

```bash
tram run \
  --matrix data/genotypes.tsv \
  --groups data/phenotypes.tsv \
  --target SP_CODE \
  --group1 Group_A \
  --group2 Group_B \
  --chromosome data/chromosomes.tsv \
  --gff data/genes.gff3 \
  --database data/go_terms.tsv \
  --function data/functions.tsv \
  --repeat-fraction 0.42 \
  --rounds 100 \
  --output outputs/group-a-group-b
```

The standard analysis runs 100 replicate models, 999 null randomizations,
10,000 bp windows, and 5,000 bp steps. Use `tram run --help` to inspect and
change every analysis parameter.

### Hyperparameter selection

TRAM 1.1 selects the tuned random forest hierarchically:

1. Hyperopt primarily maximizes mean cross-validation ROC-AUC.
2. Among trials within `--roc-auc-tolerance` of the best ROC-AUC (default
   `0.001`), TRAM minimizes log loss calculated from pooled out-of-fold
   probabilities.
3. If log loss is also within `--log-loss-tolerance` (default `0.001`), TRAM
   prefers the more regularized forest: larger `min_samples_leaf`, smaller
   `max_depth`, fewer features per split, then larger `min_samples_split`.
   Tree count is considered last.

Each tuning run writes `rf_tuned_model.tuning.json` with the policy, metrics,
parameters, and complexity key for every trial, plus the selected trial.

The three stages can also be executed independently:

```bash
tram tune --help
tram model --help
tram sliding-window --help
```

## Input Files

All tables are tab-separated.

### SNP matrix

Rows are SNPs and columns are sample IDs. `ACC` contains identifiers ending in
`_<integer position>`; everything before the final underscore is treated as the
scaffold name.

```text
ACC                 sample01  sample02
scaffold_1_100      0         1
scaffold_1_250      2         0
```

### Phenotypes

`ID` must match matrix sample columns. The selected target column contains the
two group values.

```text
ID        SP_CODE
sample01  Group_A
sample02  Group_B
```

### Chromosomes and scaffolds

```text
seqid       end      CHR
scaffold_1  1000000  CHR1
```

### Gene annotations

The GFF3 must contain `gene` records with an `ID` attribute. Coordinates and
scaffold names must use the same reference as the SNP matrix.

### Functional annotations

The functional table requires `#query`, `Description`, `GOs`, `KEGG_ko`, and
`PFAMs`. EggNOG-mapper annotation tables satisfy this layout. Gene IDs in
`#query` may include a transcript suffix.

### GO database

The GO table requires an `id` column and may contain additional descriptive
columns such as `name` and `namespace`.

## Outputs

Each complete run writes:

- `rf_tuned_model.model` and `.loss`: selected model parameters and negative
  mean CV ROC-AUC.
- `rf_tuned_model.tuning.json`: hierarchical selection policy and complete
  per-trial tuning audit.
- `<comparison>/replicates/`: model reports and SNP importance summaries.
- `sliding_window_<comparison>/`: windows, plot, threshold, genes, and GO terms.
- `run_metadata.json`: parameters, software version, timestamp, and output paths.

## Reproducibility

TRAM records run parameters and uses deterministic seeds for data splitting,
random forests, hyperparameter search, and null randomization. Record the TRAM
release, Python environment, input checksums, reference genome version, and
annotation versions in any publication.

## Memory requirements

The repeated-model stage keeps one reusable sparse one-hot matrix in memory
and creates sparse train/test slices for each replicate. Memory therefore
scales with the number of selected samples, SNPs, and observed genotype
categories. A benchmark-based estimate for 321 samples and 5,155,710 SNPs is
approximately 47--50 GB peak memory; allocate at least 64 GB for a dataset of
that size. Smaller datasets require proportionally less memory.

The randomized-window matrix is held in a temporary disk-backed array, so large
genomes require sufficient temporary storage in the output filesystem.

## Development

```bash
python -m pip install -e ".[test]"
pytest
```

Pull requests should include tests for changed behavior. Please open an issue
before changing statistical defaults or output schemas.

## Citation

Citation information is available through GitHub’s **Cite this repository**
button and in [`CITATION.cff`](CITATION.cff).

## License

TRAM is distributed under the Apache License 2.0. See [`LICENSE`](LICENSE).
