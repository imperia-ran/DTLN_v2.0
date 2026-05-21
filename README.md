# moneyprint_automaticly_DYL

`moneyprint_automaticly_DYL` is a ground-up rewrite of the original DTLN
codebase and the existing repository-specific AEC branch. The project keeps the
core Dual-Signal Transformation LSTM ideas, but turns the repository into a
maintainable Python package with explicit configuration, reusable data
pipelines, test coverage, conversion helpers, and compatibility shims for the
legacy scripts.

## What changed

- The single-file training script was decomposed into a package under
  `src/moneyprint_dtln`.
- Both the original single-input denoising model and the repository-specific
  dual-input acoustic echo cancellation model are supported.
- Dataset indexing, file pairing, frame chunking, and metadata validation now
  live in dedicated modules.
- Export to SavedModel and TFLite is routed through reusable service classes.
- Offline enhancement and frame-by-frame streaming inference are built on top of
  shared runtime code.
- The previous `DTLNbeta` scripts are kept as thin compatibility wrappers so
  existing workflows do not break immediately.

## Layout

```text
src/moneyprint_dtln/
  config.py            typed project configuration
  data/                manifests, file pairing, and tf.data creation
  models/              DTLN denoiser and AEC model builders
  training.py          trainer and callbacks
  exporters.py         SavedModel and TFLite export
  runtime.py           offline and streaming inference
  cli.py               command line entry point
tests/
  lightweight tests for config, manifests, and audio framing helpers
DTLNbeta/
  legacy compatibility wrappers
```

## Quick start

Install the package in editable mode:

```bash
pip install -e .
```

Inspect the available commands:

```bash
python -m moneyprint_dtln.cli --help
```

Create a dataset manifest:

```bash
python -m moneyprint_dtln.cli manifest \
  --root /path/to/dataset \
  --mode aec \
  --output dataset_manifest.json
```

Train the AEC model:

```bash
python -m moneyprint_dtln.cli train \
  --config configs/aec_train.json
```

Enhance a folder offline:

```bash
python -m moneyprint_dtln.cli enhance \
  --config configs/aec_infer.json \
  --input /path/to/input \
  --output /path/to/output
```

## Data conventions

The denoising mode expects parallel noisy and clean `.wav` files with the same
relative path under separate roots.

The AEC mode expects triplets that can be identified by a shared suffix:

- `nearend_mic_fileid_<id>.wav`
- `farend_speech_fileid_<id>.wav`
- `nearend_speech_fileid_<id>.wav`

The pairing logic is configurable if your naming rules differ.

## Notes

- TensorFlow is loaded lazily so utility modules and tests can run without a
  full training environment.
- The repository still contains historical model artifacts under `DTLNbeta/`.
  They are not required for the new package layout.
