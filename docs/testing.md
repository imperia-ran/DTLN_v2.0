# Testing Guide

The rewrite adds automated tests so changes can be verified without relying on
manual listening or ad hoc scripts.

## Run Tests

Install development dependencies:

```bash
pip install -e .[dev]
```

Run the suite:

```bash
python -m pytest -q
```

Expected result for the current checked-in suite:

```text
56 passed
```

## What the Tests Cover

The suite verifies the workflows that are most likely to break during a rewrite:

- Audio helpers: reading mono audio, sample-rate validation, padding, chunking,
  overlap-add, clipping, and roundtrips.
- Configuration: invalid block sizes, missing dataset roots, mismatched modes,
  and invalid dropout values.
- Manifests: serialization, JSON writing, and manifest bundle construction.
- Pairing: denoising pairs and AEC triplets.
- Preprocessing: metadata inspection, duration summaries, validation splits,
  and issue reports.
- Metrics: MSE, MAE, SNR, SI-SDR, correlation, framewise SNR, and aggregate
  evaluation.
- Recipes: training, export, and TFLite inference configuration helpers.
- Reporting: JSON reports, Markdown reports, and key-value tables.

## Integration Coverage

Several tests intentionally cover more than one module. They check that:

- Dataset roots can be converted into manifest bundles.
- Recipes populate valid runtime, training, and export paths.
- Metrics can be rendered into human-readable reports.
- Inspection helpers produce summaries that are suitable for review.

This gives reviewers evidence that the rewrite works as a connected workflow,
not only as isolated functions.

## Safety Expectations

Runtime code should not add hidden network calls, background uploads, unrelated
file writes, or environment-specific absolute paths. Inputs, outputs, and model
paths should come from explicit configuration or command line arguments.

## TensorFlow Boundary

TensorFlow is optional for lightweight tests and utility commands. The project
uses lazy imports so data validation, metrics, reporting, and most tests remain
usable in a small review environment.
