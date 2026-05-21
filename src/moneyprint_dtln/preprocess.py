"""Dataset preprocessing and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from wavinfo import WavInfoReader

from .config import AudioConfig, DatasetConfig
from .exceptions import DatasetError
from .utils.audio import read_mono_audio
from .utils.filesystem import iter_files


@dataclass(slots=True)
class FileIssue:
    path: str
    reason: str


@dataclass(slots=True)
class FileMetadata:
    path: str
    frames: int
    sample_rate: int
    duration_seconds: float


@dataclass(slots=True)
class DatasetValidationReport:
    mode: str
    total_files: int = 0
    valid_files: int = 0
    issues: list[FileIssue] = field(default_factory=list)
    metadata: list[FileMetadata] = field(default_factory=list)

    def add_issue(self, path: Path, reason: str) -> None:
        self.issues.append(FileIssue(path=str(path), reason=reason))

    def add_metadata(self, path: Path, frames: int, sample_rate: int) -> None:
        self.metadata.append(
            FileMetadata(
                path=str(path),
                frames=frames,
                sample_rate=sample_rate,
                duration_seconds=frames / sample_rate if sample_rate else 0.0,
            )
        )
        self.valid_files += 1


def scan_audio_files(root: str | Path, pattern: str = "*.wav") -> list[Path]:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise DatasetError(f"missing directory: {base}")
    return list(iter_files(base, pattern))


def inspect_audio_tree(
    root: str | Path,
    audio: AudioConfig,
    pattern: str = "*.wav",
) -> DatasetValidationReport:
    report = DatasetValidationReport(mode="tree")
    for file_path in scan_audio_files(root, pattern):
        report.total_files += 1
        try:
            chunk = read_mono_audio(file_path, expected_sample_rate=audio.sample_rate)
            report.add_metadata(file_path, len(chunk.data), chunk.sample_rate)
        except Exception as exc:
            report.add_issue(file_path, str(exc))
    return report


def inspect_dataset_config(dataset: DatasetConfig, audio: AudioConfig) -> DatasetValidationReport:
    report = DatasetValidationReport(mode=dataset.mode)
    roots: list[Path] = []
    if dataset.mode == "denoise":
        roots = [
            Path(dataset.train_noisy_root or ""),
            Path(dataset.train_clean_root or ""),
            Path(dataset.val_noisy_root or ""),
            Path(dataset.val_clean_root or ""),
        ]
    else:
        roots = [
            Path(dataset.train_noisy_root or ""),
            Path(dataset.train_clean_root or ""),
            Path(dataset.train_farend_root or ""),
            Path(dataset.val_noisy_root or ""),
            Path(dataset.val_clean_root or ""),
            Path(dataset.val_farend_root or ""),
        ]

    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved in seen or str(root) == ".":
            continue
        seen.add(resolved)
        tree_report = inspect_audio_tree(resolved, audio, pattern=dataset.file_pattern)
        report.total_files += tree_report.total_files
        report.valid_files += tree_report.valid_files
        report.issues.extend(tree_report.issues)
        report.metadata.extend(tree_report.metadata)
    return report


def chunk_count_for_file(path: str | Path, chunk_size: int) -> int:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    info = WavInfoReader(str(Path(path).expanduser().resolve()))
    return int(info.data.frame_count // chunk_size)


def summarize_durations(metadata: Iterable[FileMetadata]) -> dict[str, float]:
    items = list(metadata)
    if not items:
        return {
            "total_seconds": 0.0,
            "mean_seconds": 0.0,
            "max_seconds": 0.0,
            "min_seconds": 0.0,
        }
    durations = [item.duration_seconds for item in items]
    return {
        "total_seconds": float(sum(durations)),
        "mean_seconds": float(sum(durations) / len(durations)),
        "max_seconds": float(max(durations)),
        "min_seconds": float(min(durations)),
    }


def validate_pair_lengths(*paths: str | Path) -> bool:
    lengths = []
    for path in paths:
        info = WavInfoReader(str(Path(path).expanduser().resolve()))
        lengths.append(int(info.data.frame_count))
    return len(set(lengths)) == 1


def detect_aec_triplets(
    mic_root: str | Path,
    farend_root: str | Path,
    clean_root: str | Path,
    dataset: DatasetConfig,
) -> list[tuple[Path, Path, Path]]:
    mic_base = Path(mic_root).expanduser().resolve()
    farend_base = Path(farend_root).expanduser().resolve()
    clean_base = Path(clean_root).expanduser().resolve()
    triplets: list[tuple[Path, Path, Path]] = []

    for mic_file in iter_files(mic_base, dataset.file_pattern):
        suffix = mic_file.stem.split(dataset.suffix_separator, maxsplit=1)
        if len(suffix) != 2:
            continue
        sample_id = suffix[-1]
        farend = farend_base / f"{dataset.farend_prefix}{sample_id}{mic_file.suffix}"
        clean = clean_base / f"{dataset.clean_prefix}{sample_id}{mic_file.suffix}"
        if farend.exists() and clean.exists():
            triplets.append((mic_file, farend, clean))
    return triplets


def validate_aec_triplets(
    mic_root: str | Path,
    farend_root: str | Path,
    clean_root: str | Path,
    dataset: DatasetConfig,
    audio: AudioConfig,
) -> DatasetValidationReport:
    report = DatasetValidationReport(mode="aec")
    triplets = detect_aec_triplets(mic_root, farend_root, clean_root, dataset)
    for mic, farend, clean in triplets:
        report.total_files += 3
        try:
            mic_chunk = read_mono_audio(mic, expected_sample_rate=audio.sample_rate)
            farend_chunk = read_mono_audio(farend, expected_sample_rate=audio.sample_rate)
            clean_chunk = read_mono_audio(clean, expected_sample_rate=audio.sample_rate)
            if len({len(mic_chunk.data), len(farend_chunk.data), len(clean_chunk.data)}) != 1:
                raise DatasetError("triplet length mismatch")
            report.add_metadata(mic, len(mic_chunk.data), mic_chunk.sample_rate)
            report.add_metadata(farend, len(farend_chunk.data), farend_chunk.sample_rate)
            report.add_metadata(clean, len(clean_chunk.data), clean_chunk.sample_rate)
        except Exception as exc:
            report.add_issue(mic, str(exc))
    return report


def split_train_validation(files: list[Path], validation_ratio: float = 0.2) -> tuple[list[Path], list[Path]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    total = len(files)
    val_count = max(1, int(total * validation_ratio)) if total else 0
    validation = sorted(files[:val_count])
    training = sorted(files[val_count:])
    return training, validation


def write_issue_report(report: DatasetValidationReport, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    lines = [
        f"mode: {report.mode}",
        f"total_files: {report.total_files}",
        f"valid_files: {report.valid_files}",
        f"issues: {len(report.issues)}",
        "",
    ]
    for issue in report.issues:
        lines.append(f"{issue.path}: {issue.reason}")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
