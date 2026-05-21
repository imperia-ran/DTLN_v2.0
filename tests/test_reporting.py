from __future__ import annotations

from pathlib import Path

from moneyprint_dtln.reporting import render_key_value_table, summarize_numeric_payload, write_dual_report, write_json_report, write_markdown_report


def test_write_json_report(tmp_path: Path) -> None:
    path = write_json_report(tmp_path / "report.json", {"a": 1, "b": 2})
    assert path.exists()
    assert '"a": 1' in path.read_text(encoding="utf-8")


def test_write_markdown_report(tmp_path: Path) -> None:
    path = write_markdown_report(tmp_path / "report.md", "Title", {"Section": {"value": 1}})
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "# Title" in text
    assert "## Section" in text


def test_render_key_value_table() -> None:
    table = render_key_value_table({"snr": 12.3, "mse": 0.1})
    assert "| key | value |" in table
    assert "| snr | 12.3 |" in table


def test_summarize_numeric_payload_filters_non_numeric() -> None:
    payload = summarize_numeric_payload({"snr": 12.0, "tag": "demo", "count": 3})
    assert payload == {"snr": 12.0, "count": 3.0}


def test_write_dual_report(tmp_path: Path) -> None:
    markdown_path, json_path = write_dual_report(tmp_path / "bundle", "Run", {"metrics": {"snr": 10.0}})
    assert markdown_path.exists()
    assert json_path.exists()
    assert markdown_path.suffix == ".md"
    assert json_path.suffix == ".json"
