from __future__ import annotations

import numpy as np

from moneyprint_dtln.metrics import correlation, evaluate, framewise_snr, mae, mse, peak_to_average, si_sdr, snr


def test_mse_and_mae_are_zero_for_identical_signals() -> None:
    signal = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    assert mse(signal, signal) == 0.0
    assert mae(signal, signal) == 0.0


def test_snr_is_higher_for_cleaner_estimate() -> None:
    reference = np.ones(1024, dtype=np.float32)
    worse = reference + 0.5
    better = reference + 0.1
    assert snr(reference, better) > snr(reference, worse)


def test_si_sdr_prefers_better_estimate() -> None:
    reference = np.sin(np.linspace(0, 6.28, 2048)).astype(np.float32)
    worse = reference + 0.5 * np.random.default_rng(0).standard_normal(reference.shape).astype(np.float32)
    better = reference + 0.1 * np.random.default_rng(1).standard_normal(reference.shape).astype(np.float32)
    assert si_sdr(reference, better) > si_sdr(reference, worse)


def test_peak_to_average_handles_empty_signal() -> None:
    assert peak_to_average(np.array([], dtype=np.float32)) == 0.0


def test_framewise_snr_returns_expected_number_of_frames() -> None:
    reference = np.ones(1024, dtype=np.float32)
    estimate = np.ones(1024, dtype=np.float32) * 0.9
    values = framewise_snr(reference, estimate, frame_size=256)
    assert len(values) == 4


def test_correlation_zero_for_constant_signals() -> None:
    reference = np.ones(16, dtype=np.float32)
    estimate = np.ones(16, dtype=np.float32)
    assert correlation(reference, estimate) == 0.0


def test_evaluate_returns_composite_result() -> None:
    reference = np.sin(np.linspace(0, 10, 2048)).astype(np.float32)
    estimate = reference * 0.95
    result = evaluate(reference, estimate, frame_size=256)
    assert result.mse >= 0.0
    assert result.mae >= 0.0
    assert result.framewise_snr_mean == result.framewise_snr_mean
