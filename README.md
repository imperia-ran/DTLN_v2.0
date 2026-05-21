

It is a ground-up rewrite of the original DTLN
codebase and the existing repository-specific AEC branch. The project keeps the
core Dual-Signal Transformation LSTM ideas, but turns the repository into a
maintainable Python package with explicit configuration, reusable data
pipelines, test coverage, conversion helpers, and compatibility shims for the
legacy scripts.

## Retained advantages

The rewrite intentionally keeps the strengths of the original project and the
existing AEC branch:

- The real-time two-stage DTLN/AEC inference structure is preserved.
- Existing TFLite model artifacts under `DTLNbeta/` remain usable.
- Legacy entry points are still available through compatibility wrappers.
- The repository-specific dual-input AEC workflow is kept instead of being
  flattened into a denoising-only layout.

What improved around those strengths:

- The code is now package-structured and configuration-driven instead of being
  centered on one large script.
- Data validation, manifest building, export, inspection, and reporting are now
  first-class modules.
- Training, export, offline inference, and streaming-style inference share
  reusable runtime code.
- Regression tests and benchmark scripts are included so changes can be checked
  instead of guessed.

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

## Benchmarks

The repository now includes benchmark scripts and reports for the rewritten
runtime path.

- Real TFLite runtime benchmark: the rewritten inference loop is about 6% to 7%
  faster than the previous loop while producing identical output for the same
  AEC `.tflite` weights.
- Quality regression checks: legacy and rewritten loops produce identical audio
  output on the tested real `.tflite` models, so the rewrite does not introduce
  an inference-quality regression by itself.
- The benchmark work also showed that the current checked-in AEC weights are the
  limiting factor for quality on the tested speech cases; the runtime rewrite
  improves execution efficiency, not the learned model quality.

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
