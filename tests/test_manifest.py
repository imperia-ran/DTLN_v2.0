from __future__ import annotations

from pathlib import Path

from moneyprint_dtln.data.manifest import ManifestBundle, load_manifest, write_manifest


def test_manifest_roundtrip(tmp_path: Path) -> None:
    bundle = ManifestBundle(
        mode="aec",
        root=str(tmp_path),
        train=[{"mic": "a.wav", "farend": "b.wav", "clean": "c.wav", "split": "train", "sample_id": "1"}],
        validation=[{"mic": "d.wav", "farend": "e.wav", "clean": "f.wav", "split": "validation", "sample_id": "2"}],
    )
    path = write_manifest(tmp_path / "manifest.json", bundle)
    loaded = load_manifest(path)
    assert loaded.mode == "aec"
    assert loaded.train[0]["sample_id"] == "1"
    assert loaded.validation[0]["sample_id"] == "2"


def test_manifest_write_creates_expected_json(tmp_path: Path) -> None:
    bundle = ManifestBundle(mode="denoise", root=str(tmp_path), train=[{"noisy": "x.wav", "clean": "y.wav", "split": "train"}], validation=[])
    path = write_manifest(tmp_path / "manifest.json", bundle)
    text = path.read_text(encoding="utf-8")
    assert '"mode": "denoise"' in text
    assert '"noisy": "x.wav"' in text
