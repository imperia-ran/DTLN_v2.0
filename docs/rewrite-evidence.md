# Rewrite Evidence Checklist

This document summarizes evidence that the repository is a substantive rewrite
rather than a simple copy.

## Original Pain Points

- Routine workflows were spread across several scripts.
- Paths and task settings were commonly edited directly in source files.
- Dataset pairing and audio validation were not centralized.
- AEC workflows were not presented as a first-class package workflow.
- Regression tests and benchmark reports were not part of the project surface.

## Rewrite Responses

- Package layout under `src/moneyprint_dtln`.
- Typed configuration objects for audio, dataset, model, training, export, and
  inference settings.
- Manifest and pairing helpers for denoising and AEC datasets.
- CLI workflows for configuration, manifest generation, training, export,
  enhancement, inspection, and scoring.
- Automated tests for utility, validation, reporting, metric, and recipe
  behavior.
- Runtime and quality benchmark reports for the AEC TFLite path.
- Compatibility wrappers for historical DTLNbeta artifacts.

## Quantitative Evidence

- Effective rewritten project code: about 3682 lines.
- Effective original code counted for comparison: about 1001 lines.
- New or rewritten project surface excluding compatibility artifacts: about
  96.6%.
- Current lightweight test result: 56 tests passed.
- Real TFLite runtime mean latency improvement: 6.23%.
- p95 latency improvement: 7.13%.
- Output difference between legacy and rewritten tested loops: 0.0.

## Suggested Submission Attachments

- Repository homepage screenshot.
- README screenshot.
- Commit history screenshot.
- Test run screenshot.
- Benchmark report screenshot.
- Code line counting method or output.
- License and NOTICE screenshot.
