"""NumPy metrics for quick offline evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _safe_power(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.mean(np.square(signal.astype(np.float64))))


def snr(reference: np.ndarray, estimate: np.ndarray, epsilon: float = 1e-7) -> float:
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    noise = reference - estimate
    return 10.0 * math.log10((_safe_power(reference) + epsilon) / (_safe_power(noise) + epsilon))


def mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    if reference.size == 0:
        return 0.0
    return float(np.mean(np.square(reference - estimate)))


def mae(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    if reference.size == 0:
        return 0.0
    return float(np.mean(np.abs(reference - estimate)))


def si_sdr(reference: np.ndarray, estimate: np.ndarray, epsilon: float = 1e-7) -> float:
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    dot = np.sum(reference * estimate)
    denom = np.sum(reference * reference) + epsilon
    scale = dot / denom
    projection = scale * reference
    noise = estimate - projection
    return 10.0 * math.log10((np.sum(projection**2) + epsilon) / (np.sum(noise**2) + epsilon))


def peak_to_average(signal: np.ndarray, epsilon: float = 1e-7) -> float:
    signal = signal.astype(np.float64)
    if signal.size == 0:
        return 0.0
    peak = float(np.max(np.abs(signal)))
    avg = float(np.mean(np.abs(signal)))
    return peak / (avg + epsilon)


def framewise_snr(reference: np.ndarray, estimate: np.ndarray, frame_size: int) -> list[float]:
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    length = min(len(reference), len(estimate))
    values: list[float] = []
    for start in range(0, length - frame_size + 1, frame_size):
        stop = start + frame_size
        values.append(snr(reference[start:stop], estimate[start:stop]))
    return values


def correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    if reference.size == 0 or estimate.size == 0:
        return 0.0
    if np.std(reference) == 0 or np.std(estimate) == 0:
        return 0.0
    return float(np.corrcoef(reference, estimate)[0, 1])


@dataclass(slots=True)
class EvaluationResult:
    snr: float
    si_sdr: float
    mse: float
    mae: float
    correlation: float
    framewise_snr_mean: float


def evaluate(reference: np.ndarray, estimate: np.ndarray, frame_size: int = 512) -> EvaluationResult:
    framewise = framewise_snr(reference, estimate, frame_size=frame_size)
    framewise_mean = float(np.mean(framewise)) if framewise else 0.0
    return EvaluationResult(
        snr=snr(reference, estimate),
        si_sdr=si_sdr(reference, estimate),
        mse=mse(reference, estimate),
        mae=mae(reference, estimate),
        correlation=correlation(reference, estimate),
        framewise_snr_mean=framewise_mean,
    )
