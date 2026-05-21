"""Signal transforms shared by model construction and runtime inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .tf_compat import require_tensorflow


@dataclass(slots=True)
class FrameFeatures:
    magnitude: Any
    phase: Any


class SignalTransform:
    """Encapsulates frame/FFT/iFFT operations for consistency."""

    def __init__(self, block_len: int, block_shift: int) -> None:
        self.block_len = block_len
        self.block_shift = block_shift

    def frame(self, signal: Any) -> Any:
        tf = require_tensorflow()
        return tf.signal.frame(signal, self.block_len, self.block_shift)

    def stft(self, signal: Any) -> FrameFeatures:
        tf = require_tensorflow()
        frames = self.frame(signal)
        spectrum = tf.signal.rfft(frames)
        return FrameFeatures(magnitude=tf.abs(spectrum), phase=tf.math.angle(spectrum))

    def fft_frame(self, signal: Any) -> FrameFeatures:
        tf = require_tensorflow()
        frames = tf.expand_dims(signal, axis=1)
        spectrum = tf.signal.rfft(frames)
        return FrameFeatures(magnitude=tf.abs(spectrum), phase=tf.math.angle(spectrum))

    def ifft(self, magnitude: Any, phase: Any) -> Any:
        tf = require_tensorflow()
        complex_spectrum = tf.cast(magnitude, tf.complex64) * tf.exp(1j * tf.cast(phase, tf.complex64))
        return tf.signal.irfft(complex_spectrum)

    def overlap_add(self, frames: Any) -> Any:
        tf = require_tensorflow()
        return tf.signal.overlap_and_add(frames, self.block_shift)

    def numpy_rfft(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spectrum = np.fft.rfft(np.squeeze(frame)).astype(np.complex64)
        return np.abs(spectrum).astype(np.float32), np.angle(spectrum).astype(np.float32)

    def numpy_irfft(self, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        complex_spectrum = magnitude * np.exp(1j * phase)
        return np.fft.irfft(complex_spectrum).astype(np.float32)
