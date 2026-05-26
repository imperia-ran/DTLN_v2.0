# Architecture Overview

DTLN_v2.0 reorganizes the original script-centered workflow into a package.
The package boundaries are designed around user workflows rather than around a
single training file.

## Main Boundaries

```text
src/moneyprint_dtln/
  config.py       typed project configuration
  data/           pairing, manifests, and dataset creation
  models/         denoising and AEC model builders
  training.py     training orchestration
  exporters.py    SavedModel and TFLite export
  runtime.py      offline and streaming-style inference
  metrics.py      quality and regression metrics
  reporting.py    JSON and Markdown reports
  cli.py          command line entry point
```

## Why This Layout

The upstream DTLN repository is effective as a research implementation, but
routine usage requires navigating multiple scripts and editing paths directly.
The rewrite separates responsibilities so users can:

- Validate data before training.
- Reuse the same configuration across training, export, and inference.
- Test utility behavior without TensorFlow.
- Keep legacy model artifacts usable through compatibility wrappers.
- Add new workflow checks without modifying model-building code.

## Compatibility Layer

`DTLNbeta/` remains available for historical artifacts and compatibility
wrappers. New code should prefer `src/moneyprint_dtln`, but compatibility is
kept so existing model files can still be inspected and benchmarked.

## Review Surface

For reviewers, the most important files are:

- `README.md` for project intent and quick start.
- `docs/usage.md` for command workflows.
- `docs/data.md` for dataset expectations.
- `docs/testing.md` for verification scope.
- `docs/benchmarking.md` for runtime evidence.
- `tests/` for automated checks.
- `benchmark_reports/` for checked-in evidence.
