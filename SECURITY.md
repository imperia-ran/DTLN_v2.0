# Security Policy

DTLN_v2.0 processes local audio files and local model artifacts. Security
review should focus on explicit file access, dependency usage, and whether a
workflow introduces unexpected network or system behavior.

## Supported Version

The `main` branch is the active development branch for the current rewrite.

## Reporting a Concern

Open a private report with the repository maintainer or file an issue that
does not expose sensitive data. Include:

- The command that triggered the behavior.
- The configuration file used.
- The input and output paths involved.
- Whether TensorFlow or TFLite runtime dependencies were installed.
- Any unexpected file writes, network requests, or command execution.

## Expected Behavior

The project should:

- Read only user-provided audio, config, model, and manifest paths.
- Write only user-provided output, report, checkpoint, or export paths.
- Avoid background uploads and hidden network calls.
- Keep training and export dependencies optional for lightweight review.

## Dependency Notes

Core utilities depend on common Python packages such as `numpy`, `soundfile`,
and `wavinfo`. TensorFlow is optional and loaded lazily for training, export,
or model-building workflows.
