# Contributing

Thank you for improving DTLN_v2.0. This repository is focused on a
maintainable rewrite of the DTLN denoising and acoustic echo cancellation
workflow.

## Contribution Goals

Good contributions should make the project easier to use, easier to verify, or
safer to extend. Examples include:

- Clearer dataset validation and error messages.
- Better training, export, or inference documentation.
- Regression tests for audio framing, metrics, recipes, or reports.
- Benchmarks that compare old and rewritten behavior.
- Compatibility fixes for existing DTLNbeta model artifacts.

## Development Setup

Install the package in editable mode:

```bash
pip install -e .[dev]
```

Run the test suite before opening a pull request:

```bash
python -m pytest -q
```

Training and export paths load TensorFlow lazily. Lightweight utilities and
tests should continue to work without requiring a full training environment.

## Pull Request Checklist

- Explain the user-facing problem being solved.
- Include tests for behavior changes when practical.
- Update documentation when commands, data conventions, or outputs change.
- Keep compatibility with the checked-in DTLNbeta artifacts unless the change
  explicitly documents a migration.
- Avoid adding network calls, hidden file writes, or environment-specific paths
  to runtime code.

## Reporting Issues

When reporting a bug, include:

- The command or workflow that failed.
- The configuration file used.
- A short description of the dataset layout.
- The full error message.
- Whether TensorFlow is installed in the environment.
