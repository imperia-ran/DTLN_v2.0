"""Helpers for writing human-readable reports from inspections and evaluations."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _normalize_payload(payload: Any) -> Any:
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, dict):
        return {key: _normalize_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_normalize_payload(item) for item in payload]
    return payload


def write_json_report(path: str | Path, payload: Any) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_normalize_payload(payload), indent=2), encoding="utf-8")
    return target


def write_markdown_report(path: str | Path, title: str, sections: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for heading, body in sections.items():
        lines.append(f"## {heading}")
        lines.append("")
        if isinstance(body, dict):
            for key, value in body.items():
                lines.append(f"- `{key}`: {value}")
        elif isinstance(body, list):
            for item in body:
                lines.append(f"- {item}")
        else:
            lines.append(str(body))
        lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def render_key_value_table(payload: dict[str, Any]) -> str:
    header = ["| key | value |", "| --- | --- |"]
    rows = [f"| {key} | {value} |" for key, value in payload.items()]
    return "\n".join(header + rows)


def summarize_numeric_payload(payload: dict[str, Any]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            summary[key] = float(value)
    return summary


def write_dual_report(base_path: str | Path, title: str, sections: dict[str, Any]) -> tuple[Path, Path]:
    base = Path(base_path).expanduser().resolve()
    markdown_path = base.with_suffix(".md")
    json_path = base.with_suffix(".json")
    write_markdown_report(markdown_path, title, sections)
    write_json_report(json_path, sections)
    return markdown_path, json_path
