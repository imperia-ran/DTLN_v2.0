"""Benchmark legacy and rewritten AEC runtime loops.

Two execution modes are supported:

- `fake`: isolates Python/Numpy orchestration overhead with deterministic fake
  interpreters.
- `real`: runs the repository's real TFLite AEC models and compares the legacy
  loop with a cleaned-up rewritten loop that preserves the same tensor contract.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


BLOCK_LEN = 512
BLOCK_SHIFT = 256
FFT_BINS = BLOCK_LEN // 2 + 1
LSTM_LAYERS = 2
LSTM_UNITS = 64


def generate_signals(seconds: float, sample_rate: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    total = int(seconds * sample_rate)
    t = np.arange(total, dtype=np.float32) / sample_rate
    mic = (
        0.35 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.025 * np.random.default_rng(7).standard_normal(total).astype(np.float32)
    ).astype(np.float32)
    lpb = (
        0.30 * np.sin(2.0 * np.pi * 330.0 * t + 0.15)
        + 0.10 * np.sin(2.0 * np.pi * 660.0 * t)
        + 0.020 * np.random.default_rng(11).standard_normal(total).astype(np.float32)
    ).astype(np.float32)
    return mic, lpb


def pad_signal(signal: np.ndarray) -> np.ndarray:
    padding = np.zeros(BLOCK_LEN - BLOCK_SHIFT, dtype=np.float32)
    return np.concatenate((padding, signal.astype(np.float32, copy=False), padding))


class FakeLegacyStageOne:
    def __init__(self) -> None:
        self.inputs = [
            {"index": 0, "shape": (1, 1, FFT_BINS)},
            {"index": 1, "shape": (1, 1, FFT_BINS)},
            {"index": 2, "shape": (1, LSTM_LAYERS, LSTM_UNITS, 2)},
        ]
        self.outputs = [
            {"index": 0, "shape": (1, LSTM_LAYERS, LSTM_UNITS, 2)},
            {"index": 1, "shape": (1, 1, FFT_BINS)},
        ]
        self.tensors: dict[int, np.ndarray] = {}

    def get_input_details(self) -> list[dict[str, object]]:
        return self.inputs

    def get_output_details(self) -> list[dict[str, object]]:
        return self.outputs

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        self.tensors[index] = np.array(value, copy=True)

    def invoke(self) -> None:
        mic_mag = self.tensors[0]
        farend_mag = self.tensors[1]
        states = self.tensors[2]
        mask = np.clip((0.82 * mic_mag) - (0.05 * farend_mag), 0.0, None).astype(np.float32)
        self.tensors[100] = (states + 0.001).astype(np.float32)
        self.tensors[101] = mask

    def get_tensor(self, index: int) -> np.ndarray:
        if index == 0:
            return self.tensors[100]
        if index == 1:
            return self.tensors[101]
        raise KeyError(index)


class FakeLegacyStageTwo:
    def __init__(self) -> None:
        self.inputs = [
            {"index": 0, "shape": (1, 1, BLOCK_LEN)},
            {"index": 1, "shape": (1, LSTM_LAYERS, LSTM_UNITS, 2)},
            {"index": 2, "shape": (1, 1, BLOCK_LEN)},
        ]
        self.outputs = [
            {"index": 0, "shape": (1, LSTM_LAYERS, LSTM_UNITS, 2)},
            {"index": 1, "shape": (1, 1, BLOCK_LEN)},
        ]
        self.tensors: dict[int, np.ndarray] = {}

    def get_input_details(self) -> list[dict[str, object]]:
        return self.inputs

    def get_output_details(self) -> list[dict[str, object]]:
        return self.outputs

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        self.tensors[index] = np.array(value, copy=True)

    def invoke(self) -> None:
        estimated = self.tensors[0]
        states = self.tensors[1]
        farend = self.tensors[2]
        out = (0.90 * estimated - 0.03 * farend).astype(np.float32)
        self.tensors[200] = (states + 0.001).astype(np.float32)
        self.tensors[201] = out

    def get_tensor(self, index: int) -> np.ndarray:
        if index == 0:
            return self.tensors[200]
        if index == 1:
            return self.tensors[201]
        raise KeyError(index)


class FakeNewStageOne:
    def __init__(self) -> None:
        self.inputs = [
            {"index": 0, "shape": (1, 1, FFT_BINS)},
            {"index": 1, "shape": (1, 1, FFT_BINS)},
            {"index": 2, "shape": (1, LSTM_LAYERS, LSTM_UNITS, 2)},
        ]
        self.outputs = [
            {"index": 0, "shape": (1, 1, FFT_BINS)},
            {"index": 1, "shape": (1, LSTM_LAYERS, LSTM_UNITS, 2)},
        ]
        self.tensors: dict[int, np.ndarray] = {}

    def get_input_details(self) -> list[dict[str, object]]:
        return self.inputs

    def get_output_details(self) -> list[dict[str, object]]:
        return self.outputs

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        self.tensors[index] = np.array(value, copy=True)

    def invoke(self) -> None:
        mic_mag = self.tensors[0]
        farend_mag = self.tensors[1]
        states = self.tensors[2]
        masked_mag = np.clip((0.82 * mic_mag) - (0.05 * farend_mag), 0.0, None).astype(np.float32)
        self.tensors[300] = masked_mag
        self.tensors[301] = (states + 0.001).astype(np.float32)

    def get_tensor(self, index: int) -> np.ndarray:
        if index == 0:
            return self.tensors[300]
        if index == 1:
            return self.tensors[301]
        raise KeyError(index)


class FakeNewStageTwo:
    def __init__(self) -> None:
        self.inputs = [
            {"index": 0, "shape": (1, 1, BLOCK_LEN)},
            {"index": 1, "shape": (1, 1, BLOCK_LEN)},
            {"index": 2, "shape": (1, LSTM_LAYERS, LSTM_UNITS, 2)},
        ]
        self.outputs = [
            {"index": 0, "shape": (1, 1, BLOCK_LEN)},
            {"index": 1, "shape": (1, LSTM_LAYERS, LSTM_UNITS, 2)},
        ]
        self.tensors: dict[int, np.ndarray] = {}

    def get_input_details(self) -> list[dict[str, object]]:
        return self.inputs

    def get_output_details(self) -> list[dict[str, object]]:
        return self.outputs

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        self.tensors[index] = np.array(value, copy=True)

    def invoke(self) -> None:
        estimated = self.tensors[0]
        farend = self.tensors[1]
        states = self.tensors[2]
        out = (0.90 * estimated - 0.03 * farend).astype(np.float32)
        self.tensors[400] = out
        self.tensors[401] = (states + 0.001).astype(np.float32)

    def get_tensor(self, index: int) -> np.ndarray:
        if index == 0:
            return self.tensors[400]
        if index == 1:
            return self.tensors[401]
        raise KeyError(index)


def legacy_process_arrays(mic: np.ndarray, lpb: np.ndarray) -> np.ndarray:
    interpreter_1 = FakeLegacyStageOne()
    interpreter_2 = FakeLegacyStageTwo()
    mic = mic.astype(np.float32, copy=False)
    lpb = lpb.astype(np.float32, copy=False)
    length = min(len(mic), len(lpb))
    mic = pad_signal(mic[:length])
    lpb = pad_signal(lpb[:length])

    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()
    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()

    states_1 = np.zeros(input_details_1[2]["shape"], dtype=np.float32)
    states_2 = np.zeros(input_details_2[1]["shape"], dtype=np.float32)
    out_file = np.zeros(len(mic), dtype=np.float32)
    in_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    in_buffer_lpb = np.zeros(BLOCK_LEN, dtype=np.float32)
    out_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    num_blocks = (mic.shape[0] - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT

    for idx in range(num_blocks):
        in_buffer[:-BLOCK_SHIFT] = in_buffer[BLOCK_SHIFT:]
        in_buffer[-BLOCK_SHIFT:] = mic[idx * BLOCK_SHIFT : (idx * BLOCK_SHIFT) + BLOCK_SHIFT]
        in_buffer_lpb[:-BLOCK_SHIFT] = in_buffer_lpb[BLOCK_SHIFT:]
        in_buffer_lpb[-BLOCK_SHIFT:] = lpb[idx * BLOCK_SHIFT : (idx * BLOCK_SHIFT) + BLOCK_SHIFT]

        in_block_fft = np.fft.rfft(np.squeeze(in_buffer)).astype(np.complex64)
        in_phase = np.angle(in_block_fft)
        in_mag = np.reshape(np.abs(in_block_fft), (1, 1, -1)).astype(np.float32)
        lpb_block_fft = np.fft.rfft(np.squeeze(in_buffer_lpb)).astype(np.complex64)
        lpb_mag = np.reshape(np.abs(lpb_block_fft), (1, 1, -1)).astype(np.float32)

        interpreter_1.set_tensor(input_details_1[0]["index"], in_mag)
        interpreter_1.set_tensor(input_details_1[1]["index"], lpb_mag)
        interpreter_1.set_tensor(input_details_1[2]["index"], states_1)
        interpreter_1.invoke()
        out_mask = interpreter_1.get_tensor(output_details_1[1]["index"])
        states_1 = interpreter_1.get_tensor(output_details_1[0]["index"])

        estimated_complex = np.squeeze(out_mask) * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
        estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype(np.float32)
        in_lpb = np.reshape(in_buffer_lpb, (1, 1, -1)).astype(np.float32)

        interpreter_2.set_tensor(input_details_2[1]["index"], states_2)
        interpreter_2.set_tensor(input_details_2[0]["index"], estimated_block)
        interpreter_2.set_tensor(input_details_2[2]["index"], in_lpb)
        interpreter_2.invoke()
        out_block = interpreter_2.get_tensor(output_details_2[1]["index"])
        states_2 = interpreter_2.get_tensor(output_details_2[0]["index"])

        out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
        out_buffer[-BLOCK_SHIFT:] = 0.0
        out_buffer += np.squeeze(out_block)
        out_file[idx * BLOCK_SHIFT : (idx * BLOCK_SHIFT) + BLOCK_SHIFT] = out_buffer[:BLOCK_SHIFT]

    return out_file[(BLOCK_LEN - BLOCK_SHIFT) : (BLOCK_LEN - BLOCK_SHIFT) + length]


def rewritten_process_arrays(mic: np.ndarray, lpb: np.ndarray) -> np.ndarray:
    stage_one = FakeNewStageOne()
    stage_two = FakeNewStageTwo()
    mic = mic.astype(np.float32, copy=False)
    lpb = lpb.astype(np.float32, copy=False)
    length = min(len(mic), len(lpb))
    mic = pad_signal(mic[:length])
    lpb = pad_signal(lpb[:length])

    input_one = stage_one.get_input_details()
    output_one = stage_one.get_output_details()
    input_two = stage_two.get_input_details()
    output_two = stage_two.get_output_details()

    states_one = np.zeros(input_one[2]["shape"], dtype=np.float32)
    states_two = np.zeros(input_two[2]["shape"], dtype=np.float32)
    output = np.zeros(len(mic), dtype=np.float32)
    mic_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    lpb_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    out_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    total_blocks = (mic.shape[0] - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT

    for idx in range(total_blocks):
        start = idx * BLOCK_SHIFT
        stop = start + BLOCK_SHIFT
        mic_buffer[:-BLOCK_SHIFT] = mic_buffer[BLOCK_SHIFT:]
        mic_buffer[-BLOCK_SHIFT:] = mic[start:stop]
        lpb_buffer[:-BLOCK_SHIFT] = lpb_buffer[BLOCK_SHIFT:]
        lpb_buffer[-BLOCK_SHIFT:] = lpb[start:stop]

        mic_spectrum = np.fft.rfft(mic_buffer).astype(np.complex64)
        lpb_spectrum = np.fft.rfft(lpb_buffer).astype(np.complex64)
        mic_phase = np.angle(mic_spectrum)
        mic_mag = np.reshape(np.abs(mic_spectrum), (1, 1, -1)).astype(np.float32)
        lpb_mag = np.reshape(np.abs(lpb_spectrum), (1, 1, -1)).astype(np.float32)

        stage_one.set_tensor(input_one[0]["index"], mic_mag)
        stage_one.set_tensor(input_one[1]["index"], lpb_mag)
        stage_one.set_tensor(input_one[2]["index"], states_one)
        stage_one.invoke()
        masked_mag = stage_one.get_tensor(output_one[0]["index"])
        states_one = stage_one.get_tensor(output_one[1]["index"])

        estimated_frame = np.fft.irfft(np.squeeze(masked_mag) * np.exp(1j * mic_phase)).astype(np.float32)
        estimated_frame = np.reshape(estimated_frame, (1, 1, -1)).astype(np.float32)
        farend_frame = np.reshape(lpb_buffer, (1, 1, -1)).astype(np.float32)

        stage_two.set_tensor(input_two[0]["index"], estimated_frame)
        stage_two.set_tensor(input_two[1]["index"], farend_frame)
        stage_two.set_tensor(input_two[2]["index"], states_two)
        stage_two.invoke()
        out_block = stage_two.get_tensor(output_two[0]["index"])
        states_two = stage_two.get_tensor(output_two[1]["index"])

        out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
        out_buffer[-BLOCK_SHIFT:] = 0.0
        out_buffer += np.squeeze(out_block)
        output[start:stop] = out_buffer[:BLOCK_SHIFT]

    return output[(BLOCK_LEN - BLOCK_SHIFT) : (BLOCK_LEN - BLOCK_SHIFT) + length]


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    rtf_x: float


def create_interpreters(prefix: Path):
    import tensorflow as tf

    stage_one = tf.lite.Interpreter(model_path=str(prefix.parent / f"{prefix.name}_1.tflite"))
    stage_two = tf.lite.Interpreter(model_path=str(prefix.parent / f"{prefix.name}_2.tflite"))
    stage_one.allocate_tensors()
    stage_two.allocate_tensors()
    return stage_one, stage_two


def benchmark(name: str, fn, mic: np.ndarray, lpb: np.ndarray, repeats: int, seconds: float) -> tuple[BenchmarkResult, np.ndarray]:
    timings: list[float] = []
    output = np.array([], dtype=np.float32)
    fn(mic, lpb)
    for _ in range(repeats):
        start = time.perf_counter()
        output = fn(mic, lpb)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed_ms)
    p95_index = max(0, min(len(timings) - 1, int(round((len(timings) - 1) * 0.95))))
    p95_value = sorted(timings)[p95_index]
    result = BenchmarkResult(
        name=name,
        iterations=repeats,
        mean_ms=float(statistics.mean(timings)),
        median_ms=float(statistics.median(timings)),
        p95_ms=float(p95_value),
        rtf_x=float(seconds * 1000.0 / statistics.mean(timings)),
    )
    return result, output


def percent_delta(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark old vs rewritten AEC runtime orchestration.")
    parser.add_argument("--mode", choices=["fake", "real"], default="real", help="benchmark mode")
    parser.add_argument("--seconds", type=float, default=30.0, help="signal duration to process")
    parser.add_argument("--repeats", type=int, default=8, help="number of benchmark repetitions")
    parser.add_argument(
        "--model-prefix",
        type=Path,
        default=Path("DTLNbeta/dtln_aecmodel_blockLen512_blockShift_256"),
        help="AEC TFLite model prefix without _1/_2 suffix",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_reports/aec_runtime_benchmark.json"),
        help="where to write the JSON report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mic, lpb = generate_signals(args.seconds)
    if args.mode == "fake":
        legacy_fn = legacy_process_arrays
        new_fn = rewritten_process_arrays
        benchmark_type = "synthetic_fake_interpreter"
    else:
        legacy_fn = lambda m, f: legacy_process_arrays_real(m, f, args.model_prefix)
        new_fn = lambda m, f: rewritten_process_arrays_real(m, f, args.model_prefix)
        benchmark_type = "real_tflite_model"

    legacy_result, legacy_output = benchmark("legacy_loop", legacy_fn, mic, lpb, args.repeats, args.seconds)
    new_result, new_output = benchmark("rewritten_loop", new_fn, mic, lpb, args.repeats, args.seconds)

    diff = legacy_output - new_output
    payload = {
        "benchmark_type": benchmark_type,
        "seconds": args.seconds,
        "repeats": args.repeats,
        "model_prefix": str(args.model_prefix),
        "legacy": asdict(legacy_result),
        "rewritten": asdict(new_result),
        "comparison": {
            "mean_ms_delta_percent": percent_delta(legacy_result.mean_ms, new_result.mean_ms),
            "median_ms_delta_percent": percent_delta(legacy_result.median_ms, new_result.median_ms),
            "p95_ms_delta_percent": percent_delta(legacy_result.p95_ms, new_result.p95_ms),
            "rtf_delta_percent": percent_delta(legacy_result.rtf_x, new_result.rtf_x),
            "max_abs_output_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
            "mean_abs_output_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        },
        "notes": [
            "Both paths process the same synthetic audio length with the same block geometry.",
            "In `real` mode, both paths use the repository's existing TFLite models.",
            "Output differences should be zero or numerically tiny because the rewritten loop preserves the tensor contract.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def legacy_process_arrays_real(mic: np.ndarray, lpb: np.ndarray, model_prefix: Path) -> np.ndarray:
    interpreter_1, interpreter_2 = create_interpreters(model_prefix)
    mic = mic.astype(np.float32, copy=False)
    lpb = lpb.astype(np.float32, copy=False)
    length = min(len(mic), len(lpb))
    mic = pad_signal(mic[:length])
    lpb = pad_signal(lpb[:length])

    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()
    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()
    states_1 = np.zeros(input_details_1[2]["shape"], dtype=np.float32)
    states_2 = np.zeros(input_details_2[1]["shape"], dtype=np.float32)
    out_file = np.zeros(len(mic), dtype=np.float32)
    in_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    in_buffer_lpb = np.zeros(BLOCK_LEN, dtype=np.float32)
    out_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    num_blocks = (mic.shape[0] - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT

    for idx in range(num_blocks):
        in_buffer[:-BLOCK_SHIFT] = in_buffer[BLOCK_SHIFT:]
        in_buffer[-BLOCK_SHIFT:] = mic[idx * BLOCK_SHIFT : (idx * BLOCK_SHIFT) + BLOCK_SHIFT]
        in_buffer_lpb[:-BLOCK_SHIFT] = in_buffer_lpb[BLOCK_SHIFT:]
        in_buffer_lpb[-BLOCK_SHIFT:] = lpb[idx * BLOCK_SHIFT : (idx * BLOCK_SHIFT) + BLOCK_SHIFT]

        in_block_fft = np.fft.rfft(np.squeeze(in_buffer)).astype(np.complex64)
        in_phase = np.angle(in_block_fft)
        in_mag = np.reshape(np.abs(in_block_fft), (1, 1, -1)).astype(np.float32)
        lpb_block_fft = np.fft.rfft(np.squeeze(in_buffer_lpb)).astype(np.complex64)
        lpb_mag = np.reshape(np.abs(lpb_block_fft), (1, 1, -1)).astype(np.float32)

        interpreter_1.set_tensor(input_details_1[0]["index"], in_mag)
        interpreter_1.set_tensor(input_details_1[1]["index"], lpb_mag)
        interpreter_1.set_tensor(input_details_1[2]["index"], states_1)
        interpreter_1.invoke()
        out_mask = interpreter_1.get_tensor(output_details_1[1]["index"])
        states_1 = interpreter_1.get_tensor(output_details_1[0]["index"])

        estimated_complex = np.squeeze(out_mask) * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
        estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype(np.float32)
        in_lpb = np.reshape(in_buffer_lpb, (1, 1, -1)).astype(np.float32)

        interpreter_2.set_tensor(input_details_2[1]["index"], states_2)
        interpreter_2.set_tensor(input_details_2[0]["index"], estimated_block)
        interpreter_2.set_tensor(input_details_2[2]["index"], in_lpb)
        interpreter_2.invoke()
        out_block = interpreter_2.get_tensor(output_details_2[1]["index"])
        states_2 = interpreter_2.get_tensor(output_details_2[0]["index"])

        out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
        out_buffer[-BLOCK_SHIFT:] = 0.0
        out_buffer += np.squeeze(out_block)
        out_file[idx * BLOCK_SHIFT : (idx * BLOCK_SHIFT) + BLOCK_SHIFT] = out_buffer[:BLOCK_SHIFT]

    return out_file[(BLOCK_LEN - BLOCK_SHIFT) : (BLOCK_LEN - BLOCK_SHIFT) + length]


def rewritten_process_arrays_real(mic: np.ndarray, lpb: np.ndarray, model_prefix: Path) -> np.ndarray:
    stage_one, stage_two = create_interpreters(model_prefix)
    mic = mic.astype(np.float32, copy=False)
    lpb = lpb.astype(np.float32, copy=False)
    length = min(len(mic), len(lpb))
    mic = pad_signal(mic[:length])
    lpb = pad_signal(lpb[:length])

    input_one = stage_one.get_input_details()
    output_one = stage_one.get_output_details()
    input_two = stage_two.get_input_details()
    output_two = stage_two.get_output_details()

    state_one_index = input_one[2]["index"]
    state_two_index = input_two[1]["index"]
    mic_mag_index = input_one[0]["index"]
    farend_mag_index = input_one[1]["index"]
    estimated_index = input_two[0]["index"]
    farend_frame_index = input_two[2]["index"]
    state_one_out_index = output_one[0]["index"]
    masked_mag_out_index = output_one[1]["index"]
    state_two_out_index = output_two[0]["index"]
    out_frame_index = output_two[1]["index"]

    states_one = np.zeros(input_one[2]["shape"], dtype=np.float32)
    states_two = np.zeros(input_two[1]["shape"], dtype=np.float32)
    output = np.zeros(len(mic), dtype=np.float32)
    mic_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    farend_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    out_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    total_blocks = (mic.shape[0] - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT

    for idx in range(total_blocks):
        start = idx * BLOCK_SHIFT
        stop = start + BLOCK_SHIFT
        mic_buffer[:-BLOCK_SHIFT] = mic_buffer[BLOCK_SHIFT:]
        mic_buffer[-BLOCK_SHIFT:] = mic[start:stop]
        farend_buffer[:-BLOCK_SHIFT] = farend_buffer[BLOCK_SHIFT:]
        farend_buffer[-BLOCK_SHIFT:] = lpb[start:stop]

        mic_spectrum = np.fft.rfft(mic_buffer).astype(np.complex64)
        farend_spectrum = np.fft.rfft(farend_buffer).astype(np.complex64)
        mic_phase = np.angle(mic_spectrum)
        mic_mag = np.abs(mic_spectrum).reshape(1, 1, -1).astype(np.float32, copy=False)
        farend_mag = np.abs(farend_spectrum).reshape(1, 1, -1).astype(np.float32, copy=False)

        stage_one.set_tensor(mic_mag_index, mic_mag)
        stage_one.set_tensor(farend_mag_index, farend_mag)
        stage_one.set_tensor(state_one_index, states_one)
        stage_one.invoke()
        masked_mag = stage_one.get_tensor(masked_mag_out_index)
        states_one = stage_one.get_tensor(state_one_out_index)

        estimated_frame = np.fft.irfft(np.squeeze(masked_mag) * np.exp(1j * mic_phase)).astype(np.float32)
        estimated_frame = estimated_frame.reshape(1, 1, -1)
        farend_frame = farend_buffer.reshape(1, 1, -1).astype(np.float32, copy=False)

        stage_two.set_tensor(estimated_index, estimated_frame)
        stage_two.set_tensor(state_two_index, states_two)
        stage_two.set_tensor(farend_frame_index, farend_frame)
        stage_two.invoke()
        out_block = stage_two.get_tensor(out_frame_index)
        states_two = stage_two.get_tensor(state_two_out_index)

        out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
        out_buffer[-BLOCK_SHIFT:] = 0.0
        out_buffer += np.squeeze(out_block)
        output[start:stop] = out_buffer[:BLOCK_SHIFT]

    return output[(BLOCK_LEN - BLOCK_SHIFT) : (BLOCK_LEN - BLOCK_SHIFT) + length]


if __name__ == "__main__":
    raise SystemExit(main())
