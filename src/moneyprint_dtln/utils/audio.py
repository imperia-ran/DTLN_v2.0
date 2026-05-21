"""Pure-Python audio helpers that do not depend on TensorFlow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from ..exceptions import DatasetError


@dataclass(slots=True)
class AudioChunk:
    data: np.ndarray
    sample_rate: int
    path: Path | None = None


def read_mono_audio(path: str | Path, expected_sample_rate: int | None = None) -> AudioChunk:
    file_path = Path(path)
    data, sample_rate = sf.read(str(file_path))
    if data.ndim != 1:
        raise DatasetError(f"expected mono wav file: {file_path}")
    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        raise DatasetError(
            f"sample rate mismatch for {file_path}: "
            f"expected {expected_sample_rate}, got {sample_rate}"
        )
    return AudioChunk(data=data.astype(np.float32), sample_rate=sample_rate, path=file_path)


def ensure_same_length(*signals: np.ndarray) -> tuple[np.ndarray, ...]:
    if not signals:
        return ()
    target = min(len(signal) for signal in signals)
    return tuple(signal[:target] for signal in signals)


def pad_signal(signal: np.ndarray, left: int, right: int) -> np.ndarray:
    if left < 0 or right < 0:
        raise ValueError("left and right padding must be non-negative")
    if left == 0 and right == 0:
        return signal.astype(np.float32, copy=True)
    return np.concatenate(
        (
            np.zeros(left, dtype=np.float32),
            signal.astype(np.float32, copy=False),
            np.zeros(right, dtype=np.float32),
        )
    )


def sliding_chunks(signal: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    total = len(signal) // chunk_size
    for idx in range(total):
        start = idx * chunk_size
        end = start + chunk_size
        yield signal[start:end]


def overlap_add(frames: np.ndarray, block_shift: int) -> np.ndarray:
    if frames.ndim != 2:
        raise ValueError("frames must be a 2-D array")
    if block_shift <= 0:
        raise ValueError("block_shift must be positive")
    if len(frames) == 0:
        return np.zeros(0, dtype=np.float32)

    frame_size = frames.shape[1]
    output_size = frame_size + (len(frames) - 1) * block_shift
    output = np.zeros(output_size, dtype=np.float32)
    for index, frame in enumerate(frames):
        start = index * block_shift
        output[start : start + frame_size] += frame.astype(np.float32, copy=False)
    return output


def clip_audio(signal: np.ndarray, limit: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak <= 1.0:
        return signal.astype(np.float32, copy=False)
    return (signal / peak * limit).astype(np.float32)
