"""JSON manifest types for train and validation data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ManifestMode = Literal["denoise", "aec"]


@dataclass(slots=True)
class DenoiseSample:
    noisy: str
    clean: str
    split: str


@dataclass(slots=True)
class AECSample:
    mic: str
    farend: str
    clean: str
    split: str
    sample_id: str


@dataclass(slots=True)
class ManifestBundle:
    mode: ManifestMode
    root: str
    train: list[dict[str, Any]] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_manifest(path: str | Path, bundle: ManifestBundle) -> Path:
    target = Path(path).expanduser().resolve()
    target.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    return target


def load_manifest(path: str | Path) -> ManifestBundle:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    return ManifestBundle(
        mode=payload["mode"],
        root=payload["root"],
        train=list(payload.get("train", [])),
        validation=list(payload.get("validation", [])),
    )
