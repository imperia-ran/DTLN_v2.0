from __future__ import annotations

from pathlib import Path

from moneyprint_dtln.config import AudioConfig, DatasetConfig
from moneyprint_dtln.preprocess import (
    chunk_count_for_file,
    detect_aec_triplets,
    inspect_audio_tree,
    split_train_validation,
    summarize_durations,
    validate_aec_triplets,
    validate_pair_lengths,
    write_issue_report,
)

from conftest import sine, write_wav


def test_inspect_audio_tree_collects_metadata(tmp_path: Path) -> None:
    audio = AudioConfig()
    write_wav(tmp_path / "a.wav", sine(1600))
    write_wav(tmp_path / "b.wav", sine(800))
    report = inspect_audio_tree(tmp_path, audio)
    assert report.total_files == 2
    assert report.valid_files == 2
    assert len(report.issues) == 0


def test_chunk_count_for_file(tmp_path: Path) -> None:
    wav = write_wav(tmp_path / "sample.wav", sine(16000))
    assert chunk_count_for_file(wav, 1600) == 10


def test_validate_pair_lengths(tmp_path: Path) -> None:
    left = write_wav(tmp_path / "left.wav", sine(1600))
    right = write_wav(tmp_path / "right.wav", sine(1600))
    assert validate_pair_lengths(left, right) is True


def test_validate_pair_lengths_detects_mismatch(tmp_path: Path) -> None:
    left = write_wav(tmp_path / "left.wav", sine(1600))
    right = write_wav(tmp_path / "right.wav", sine(1200))
    assert validate_pair_lengths(left, right) is False


def test_detect_aec_triplets(tmp_path: Path) -> None:
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
    triplets = detect_aec_triplets(mic_root, farend_root, clean_root, config)
    assert len(triplets) == 1


def test_validate_aec_triplets_collects_metadata(tmp_path: Path) -> None:
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
    report = validate_aec_triplets(mic_root, farend_root, clean_root, config, AudioConfig())
    assert report.valid_files == 3
    assert len(report.issues) == 0


def test_split_train_validation(tmp_path: Path) -> None:
    files = [tmp_path / f"{idx}.wav" for idx in range(10)]
    train, validation = split_train_validation(files, validation_ratio=0.2)
    assert len(train) == 8
    assert len(validation) == 2


def test_summarize_durations(tmp_path: Path) -> None:
    report = inspect_audio_tree(tmp_path, AudioConfig())
    summary = summarize_durations(report.metadata)
    assert summary["total_seconds"] == 0.0


def test_write_issue_report(tmp_path: Path) -> None:
    report = inspect_audio_tree(tmp_path, AudioConfig())
    path = write_issue_report(report, tmp_path / "issues.txt")
    assert path.exists()
