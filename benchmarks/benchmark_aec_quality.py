"""Quality benchmark for the repository's AEC TFLite models.

The benchmark synthesizes a clean near-end signal, a far-end reference, and a
microphone mixture with echo and additive noise. It then runs both the legacy
AEC loop and the rewritten AEC loop against the same real TFLite models and
scores each output against the known clean signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_aec_runtime import legacy_process_arrays_real, rewritten_process_arrays_real
from moneyprint_dtln.metrics import EvaluationResult, evaluate


def synthesize_case(seconds: float, sample_rate: int = 16000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = int(seconds * sample_rate)
    t = np.arange(total, dtype=np.float32) / sample_rate

    clean = (
        0.24 * np.sin(2.0 * np.pi * 180.0 * t)
        + 0.18 * np.sin(2.0 * np.pi * 260.0 * t + 0.3)
        + 0.10 * np.sin(2.0 * np.pi * 370.0 * t + 0.9)
    ).astype(np.float32)

    envelope = (0.65 + 0.35 * np.sin(2.0 * np.pi * 1.7 * t + 0.2)).astype(np.float32)
    clean *= envelope

    farend = (
        0.20 * np.sin(2.0 * np.pi * 220.0 * t + 0.5)
        + 0.12 * np.sin(2.0 * np.pi * 510.0 * t + 0.8)
        + 0.08 * np.sin(2.0 * np.pi * 730.0 * t + 0.1)
    ).astype(np.float32)

    delay = int(0.045 * sample_rate)
    echo = np.zeros_like(farend)
    echo[delay:] = 0.55 * farend[:-delay]
    echo += 0.15 * np.concatenate(([0.0], echo[:-1])).astype(np.float32)

    noise = 0.018 * np.random.default_rng(42).standard_normal(total).astype(np.float32)
    mic = clean + echo + noise
    return clean.astype(np.float32), farend.astype(np.float32), mic.astype(np.float32)


def serialize_result(result: EvaluationResult) -> dict[str, float]:
    return {
        "snr": result.snr,
        "si_sdr": result.si_sdr,
        "mse": result.mse,
        "mae": result.mae,
        "correlation": result.correlation,
        "framewise_snr_mean": result.framewise_snr_mean,
    }


def improvement(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in after.items():
        payload[f"{key}_delta"] = value - before[key]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality benchmark for real AEC TFLite models")
    parser.add_argument("--seconds", type=float, default=15.0, help="signal duration")
    parser.add_argument(
        "--model-prefix",
        type=Path,
        default=Path("DTLNbeta/dtln_aecmodel_blockLen512_blockShift_256"),
        help="AEC TFLite model prefix without _1/_2 suffix",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_reports/aec_quality_benchmark_real.json"),
        help="where to write the JSON report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean, farend, mic = synthesize_case(args.seconds)
    legacy = legacy_process_arrays_real(mic, farend, args.model_prefix)
    rewritten = rewritten_process_arrays_real(mic, farend, args.model_prefix)

    noisy_metrics = serialize_result(evaluate(clean, mic))
    legacy_metrics = serialize_result(evaluate(clean, legacy))
    rewritten_metrics = serialize_result(evaluate(clean, rewritten))
    loop_diff = legacy - rewritten

    payload = {
        "benchmark_type": "real_tflite_quality",
        "seconds": args.seconds,
        "model_prefix": str(args.model_prefix),
        "input_metrics": noisy_metrics,
        "legacy_metrics": legacy_metrics,
        "rewritten_metrics": rewritten_metrics,
        "legacy_improvement_over_input": improvement(noisy_metrics, legacy_metrics),
        "rewritten_improvement_over_input": improvement(noisy_metrics, rewritten_metrics),
        "loop_output_diff": {
            "max_abs": float(np.max(np.abs(loop_diff))) if loop_diff.size else 0.0,
            "mean_abs": float(np.mean(np.abs(loop_diff))) if loop_diff.size else 0.0,
        },
        "notes": [
            "The clean reference is synthetic and fully known, so the metric comparison is deterministic.",
            "Legacy and rewritten loops use the same TFLite weights; any metric gap should come from orchestration differences only.",
            "These metrics are useful for regression checks, not as a substitute for a real speech dataset evaluation.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
