# Example Analysis

The automated tests generate a small synthetic dataset demonstrating every
required input schema. For a real analysis, place your files outside the source
repository and run:

```bash
tram run \
  -m /path/to/genotypes.tsv \
  -g /path/to/phenotypes.tsv \
  -t SP_CODE \
  -1 Red \
  -2 Fodder \
  -c /path/to/chromosomes.tsv \
  --gff /path/to/genes.gff3 \
  -d /path/to/go_terms.tsv \
  -f /path/to/functional_annotations.tsv \
  -r 0.42 \
  --rounds 100 \
  -o outputs/red-fodder
```

For an initial infrastructure check, reduce runtime with `--rounds 2`,
`--replicates 2`, and `--randomizations 10`. Those reduced settings are not
intended for biological interpretation.

