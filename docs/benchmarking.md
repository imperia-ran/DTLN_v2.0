# Benchmarking Notes

DTLN_v2.0 includes benchmark scripts and checked-in reports to show whether the
rewrite changes runtime behavior.

## Runtime Benchmark

Report:

```text
benchmark_reports/aec_runtime_benchmark_real.json
```

The real TFLite runtime benchmark compares the legacy loop and the rewritten
loop using the same checked-in AEC TFLite artifacts.

Current summary:

| Metric | Legacy | Rewritten | Change |
| --- | ---: | ---: | ---: |
| Mean latency | 199.273 ms | 186.861 ms | -6.23% |
| Median latency | 200.745 ms | 187.359 ms | -6.67% |
| p95 latency | 205.147 ms | 190.513 ms | -7.13% |
| Real-time factor | 75.27x | 80.27x | +6.64% |
| Max output difference | 0.0 | 0.0 | unchanged |

## Quality Regression Report

Report:

```text
benchmark_reports/aec_quality_benchmark_real.json
```

The quality regression report compares the legacy and rewritten orchestration
paths with the same model weights. The checked-in report shows:

- `loop_output_diff.max_abs = 0.0`
- `loop_output_diff.mean_abs = 0.0`

This means the rewritten loop preserves the tested model output exactly while
improving runtime performance.

## How to Use These Reports

The reports are intended for regression evidence:

- They verify that orchestration changes did not alter output tensors.
- They provide measurable runtime differences.
- They help reviewers distinguish engineering improvements from model-quality
  claims.

They are not a substitute for a large speech dataset evaluation when training
new weights. For new model-quality claims, evaluate on a representative
dataset and include the dataset description, metrics, and scripts used.
