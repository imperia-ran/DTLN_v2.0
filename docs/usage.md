# Usage Guide

This guide describes the normal workflow for DTLN_v2.0. The goal of the
rewrite is to avoid editing source files for routine training, export, and
inference tasks.

## Install

Install the base package:

```bash
pip install -e .
```

Install development dependencies for tests:

```bash
pip install -e .[dev]
```

Install training or export dependencies when TensorFlow workflows are needed:

```bash
pip install -e .[train]
pip install -e .[export]
```

## Inspect Commands

```bash
python -m moneyprint_dtln.cli --help
```

The command line entry point exposes:

- `dump-config` for writing a default configuration template.
- `manifest` for building a dataset manifest.
- `train` for model training.
- `export` for SavedModel or TFLite export.
- `enhance` for offline enhancement.
- `inspect` for dataset and model summaries.
- `score` for evaluating one estimate against a reference.

## Create a Config Template

```bash
python -m moneyprint_dtln.cli dump-config --output config.json
```

Start from the generated file or one of the examples in `configs/`.

## Build a Dataset Manifest

```bash
python -m moneyprint_dtln.cli manifest \
  --config configs/aec_train.json \
  --root /path/to/dataset \
  --output dataset_manifest.json
```

The manifest step is useful because it turns filename assumptions into an
explicit, reviewable artifact before training.

## Train

```bash
python -m moneyprint_dtln.cli train --config configs/aec_train.json
```

Training requires TensorFlow. Utility and validation commands remain usable
without TensorFlow because the package loads it lazily.

## Export

```bash
python -m moneyprint_dtln.cli export --config configs/aec_train.json
```

The export workflow uses the model and export sections of the config file.

## Enhance Audio

```bash
python -m moneyprint_dtln.cli enhance \
  --config configs/aec_infer_tflite.json \
  --input /path/to/input \
  --output /path/to/output
```

The output directory receives enhanced audio while preserving the configured
input traversal behavior.

## Score Output

```bash
python -m moneyprint_dtln.cli score \
  --reference clean.wav \
  --estimate enhanced.wav
```

The score command reports metrics such as SNR, SI-SDR, MSE, MAE, correlation,
and framewise SNR.
