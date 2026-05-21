from __future__ import annotations

import numpy as np

from moneyprint_dtln.inspection import compare_signals, format_dataset_summary, format_model_summary


def test_compare_signals_reports_basic_statistics() -> None:
    reference = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    estimate = np.array([0.1, 0.1, 0.25], dtype=np.float32)
    report = compare_signals(reference, estimate)
    assert report["reference_power"] > 0.0
    assert report["error_power"] >= 0.0
    assert abs(report["peak_reference"] - 0.3) < 1e-6


def test_format_dataset_summary_contains_expected_keys() -> None:
    class Summary:
        mode = "aec"
        train_items = 10
        train_chunks = 100
        validation_items = 2
        validation_chunks = 20
        issue_count = 0
        duration_stats = {"total_seconds": 30.0}

    text = format_dataset_summary(Summary())
    assert "mode: aec" in text
    assert "train_items: 10" in text
    assert "total_seconds: 30.00" in text


def test_format_model_summary_contains_shapes() -> None:
    class Summary:
        name = "demo"
        parameters = 123
        input_shapes = ["(None, None)"]
        output_shapes = ["(None, None)"]

    text = format_model_summary(Summary())
    assert "name: demo" in text
    assert "parameters: 123" in text
    assert "(None, None)" in text


def test_compare_signals_handles_empty_arrays() -> None:
    report = compare_signals(np.array([], dtype=np.float32), np.array([], dtype=np.float32))
    assert report["reference_power"] == 0.0
    assert report["error_power"] == 0.0
