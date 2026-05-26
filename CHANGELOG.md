# Changelog

All notable project changes should be recorded here. The format is lightweight
and focused on what reviewers and users need to verify.

## 0.1.0 - 2026-05-26

### Added

- Rewritten package layout under `src/moneyprint_dtln`.
- Typed configuration for audio, dataset, model, training, export, and
  inference workflows.
- Denoising and acoustic echo cancellation model builders.
- Dataset pairing, manifest generation, preprocessing checks, and reporting
  helpers.
- Offline enhancement and streaming-style runtime helpers.
- Export service for SavedModel and TFLite workflows.
- Command line entry point for manifest, training, export, enhancement,
  inspection, scoring, and default config generation.
- Regression tests for audio helpers, configuration, manifests, pairing,
  preprocessing, metrics, recipes, inspection, and reporting.
- Runtime and quality benchmark reports for the AEC TFLite path.
- Compatibility wrappers and checked-in DTLNbeta artifacts.

### Improved

- Replaced script-centered usage with a package and configuration driven
  workflow.
- Added early dataset validation to catch missing files, inconsistent lengths,
  and naming mismatches.
- Preserved model output parity for the tested TFLite runtime path while
  improving runtime performance.

### Migration

- The project moved from `GR33N-WCL/DTLN_improved` to
  `imperia-ran/DTLN_v2.0`.
