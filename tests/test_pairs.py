from __future__ import annotations

from pathlib import Path

import pytest

from moneyprint_dtln.config import DatasetConfig
from moneyprint_dtln.data.pairs import build_aec_samples, build_denoise_samples, build_manifest_bundle
from moneyprint_dtln.exceptions import DatasetError

from conftest import sine, write_wav


def test_build_denoise_samples_matches_relative_pairs(tmp_path: Path) -> None:
    noisy_root = tmp_path / "noisy"
    clean_root = tmp_path / "clean"
    write_wav(noisy_root / "a.wav", sine(1600))
    write_wav(clean_root / "a.wav", sine(1600))
    write_wav(noisy_root / "nested" / "b.wav", sine(1600))
    write_wav(clean_root / "nested" / "b.wav", sine(1600))

    samples = build_denoise_samples(noisy_root, clean_root)
    assert len(samples) == 2
    assert samples[0].split == "train"
    assert samples[0].noisy.endswith(".wav")


def test_build_denoise_samples_raises_for_missing_clean_file(tmp_path: Path) -> None:
    noisy_root = tmp_path / "noisy"
    clean_root = tmp_path / "clean"
    write_wav(noisy_root / "a.wav", sine(1600))
    with pytest.raises(DatasetError):
        build_denoise_samples(noisy_root, clean_root)


def test_build_aec_samples_matches_triplets(tmp_path: Path) -> None:
    mic_root = tmp_path / "mic"
    farend_root = tmp_path / "farend"
    clean_root = tmp_path / "clean"
    write_wav(mic_root / "nearend_mic_fileid_001.wav", sine(1600))
    write_wav(farend_root / "farend_speech_fileid_001.wav", sine(1600))
    write_wav(clean_root / "nearend_speech_fileid_001.wav", sine(1600))

    config = DatasetConfig(
        mode="aec",
        train_noisy_root=mic_root,
        train_clean_root=clean_root,
        train_farend_root=farend_root,
        val_noisy_root=mic_root,
        val_clean_root=clean_root,
        val_farend_root=farend_root,
    )
    config.validate()
    samples = build_aec_samples(mic_root, farend_root, clean_root, config)
    assert len(samples) == 1
    assert samples[0].sample_id == "001"


def test_build_aec_samples_raises_for_missing_farend_or_clean(tmp_path: Path) -> None:
    mic_root = tmp_path / "mic"
    farend_root = tmp_path / "farend"
    clean_root = tmp_path / "clean"
    write_wav(mic_root / "nearend_mic_fileid_001.wav", sine(1600))

    config = DatasetConfig(
        mode="aec",
        train_noisy_root=mic_root,
        train_clean_root=clean_root,
        train_farend_root=farend_root,
        val_noisy_root=mic_root,
        val_clean_root=clean_root,
        val_farend_root=farend_root,
    )
    config.validate()
    with pytest.raises(DatasetError):
        build_aec_samples(mic_root, farend_root, clean_root, config)


def test_build_manifest_bundle_for_aec(tmp_path: Path) -> None:
    train_mic = tmp_path / "train_mic"
    train_farend = tmp_path / "train_farend"
    train_clean = tmp_path / "train_clean"
    val_mic = tmp_path / "val_mic"
    val_farend = tmp_path / "val_farend"
    val_clean = tmp_path / "val_clean"
    for root, prefix in (
        (train_mic, "nearend_mic_fileid_"),
        (train_farend, "farend_speech_fileid_"),
        (train_clean, "nearend_speech_fileid_"),
        (val_mic, "nearend_mic_fileid_"),
        (val_farend, "farend_speech_fileid_"),
        (val_clean, "nearend_speech_fileid_"),
    ):
        write_wav(root / f"{prefix}123.wav", sine(1600))

    config = DatasetConfig(
        mode="aec",
        train_noisy_root=train_mic,
        train_clean_root=train_clean,
        train_farend_root=train_farend,
        val_noisy_root=val_mic,
        val_clean_root=val_clean,
        val_farend_root=val_farend,
    )
    bundle = build_manifest_bundle(tmp_path, config)
    assert bundle.mode == "aec"
    assert len(bundle.train) == 1
    assert len(bundle.validation) == 1


def test_build_manifest_bundle_for_denoise(tmp_path: Path) -> None:
    train_noisy = tmp_path / "train_noisy"
    train_clean = tmp_path / "train_clean"
    val_noisy = tmp_path / "val_noisy"
    val_clean = tmp_path / "val_clean"
    write_wav(train_noisy / "a.wav", sine(1600))
    write_wav(train_clean / "a.wav", sine(1600))
    write_wav(val_noisy / "b.wav", sine(1600))
    write_wav(val_clean / "b.wav", sine(1600))

    config = DatasetConfig(
        mode="denoise",
        train_noisy_root=train_noisy,
        train_clean_root=train_clean,
        val_noisy_root=val_noisy,
        val_clean_root=val_clean,
    )
    bundle = build_manifest_bundle(tmp_path, config)
    assert bundle.mode == "denoise"
    assert len(bundle.train) == 1
    assert len(bundle.validation) == 1
