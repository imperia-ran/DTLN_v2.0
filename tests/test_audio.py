from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moneyprint_dtln.utils.audio import (
    clip_audio,
    ensure_same_length,
    overlap_add,
    pad_signal,
    read_mono_audio,
    sliding_chunks,
)

from conftest import sine, write_wav


def test_read_mono_audio_roundtrip(tmp_path: Path) -> None:
    wav = write_wav(tmp_path / "sample.wav", sine(1600))
    audio = read_mono_audio(wav, expected_sample_rate=16000)
    assert audio.sample_rate == 16000
    assert audio.data.ndim == 1
    assert len(audio.data) == 1600


def test_read_mono_audio_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    wav = write_wav(tmp_path / "wrong_sr.wav", sine(800), sample_rate=8000)
    with pytest.raises(Exception):
        read_mono_audio(wav, expected_sample_rate=16000)


def test_ensure_same_length_trims_to_shortest() -> None:
    left = np.arange(10, dtype=np.float32)
    right = np.arange(7, dtype=np.float32)
    clean = np.arange(5, dtype=np.float32)
    trimmed = ensure_same_length(left, right, clean)
    assert all(len(item) == 5 for item in trimmed)


def test_pad_signal_adds_left_and_right_padding() -> None:
    signal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    padded = pad_signal(signal, 2, 1)
    assert padded.tolist() == [0.0, 0.0, 1.0, 2.0, 3.0, 0.0]


def test_pad_signal_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        pad_signal(np.ones(4, dtype=np.float32), -1, 0)


def test_sliding_chunks_splits_exact_blocks() -> None:
    signal = np.arange(12, dtype=np.float32)
    chunks = list(sliding_chunks(signal, 4))
    assert len(chunks) == 3
    assert np.array_equal(chunks[0], np.array([0, 1, 2, 3], dtype=np.float32))
    assert np.array_equal(chunks[2], np.array([8, 9, 10, 11], dtype=np.float32))


def test_sliding_chunks_ignores_remainder() -> None:
    signal = np.arange(10, dtype=np.float32)
    chunks = list(sliding_chunks(signal, 4))
    assert len(chunks) == 2


def test_overlap_add_reconstructs_simple_frames() -> None:
    frames = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    reconstructed = overlap_add(frames, block_shift=2)
    assert reconstructed.tolist() == [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_overlap_add_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        overlap_add(np.ones(8, dtype=np.float32), block_shift=2)


def test_clip_audio_leaves_unit_range_untouched() -> None:
    signal = np.array([0.2, -0.5, 0.8], dtype=np.float32)
    clipped = clip_audio(signal)
    assert np.allclose(signal, clipped)


def test_clip_audio_normalizes_peaks_above_one() -> None:
    signal = np.array([0.5, 2.0, -1.5], dtype=np.float32)
    clipped = clip_audio(signal)
    assert float(np.max(np.abs(clipped))) <= 0.99 + 1e-6


def test_audio_helpers_support_longer_roundtrip(tmp_path: Path) -> None:
    wav = write_wav(tmp_path / "long.wav", sine(3200, scale=0.2, frequency=330.0))
    chunk = read_mono_audio(wav, expected_sample_rate=16000)
    padded = pad_signal(chunk.data, 128, 128)
    slices = list(sliding_chunks(padded, 256))
    stacked = np.stack(slices[:4], axis=0)
    reconstructed = overlap_add(stacked, block_shift=128)
    assert reconstructed.ndim == 1
    assert len(reconstructed) > 0
