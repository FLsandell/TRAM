# Changelog

All notable changes to TRAM will be documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-09-04

### Changed

- Hyperopt still optimizes mean cross-validation ROC-AUC as its primary loss.
- Final model selection now uses out-of-fold log loss for trials within 0.001
  ROC-AUC of the best trial, with both tolerances configurable from the CLI.
- Effectively tied metric results now prefer a more regularized random forest;
  tree count is deliberately the final complexity consideration.
- Tuning writes a JSON audit containing every trial's metrics, parameters,
  complexity key, selection tolerances, and selected trial.
- Repeated modeling now performs one-hot encoding once per invocation instead
  of repeating it for every replicate, while preserving each training split's
  category vocabulary and feature-importance mapping.
- One-hot matrices use 32-bit values, matching scikit-learn's internal random
  forest representation and limiting the additional memory retained by the
  reusable encoding.

## [1.0.0] - 2026-06-23

### Added

- Installable `tram` command with `run`, `tune`, `model`, and `sliding-window` stages.
- Reproducible random seeds and configurable scientific parameters.
- Input validation, run metadata, automated tests, and publication metadata.
- Tested support for Python 3.10 through 3.13.
- Fast CLI help through lazy loading of scientific dependencies.
- Model replicate progress bar and faster SNP importance aggregation.
- Support eggNOG annotation files with metadata lines before the `#query` header.
