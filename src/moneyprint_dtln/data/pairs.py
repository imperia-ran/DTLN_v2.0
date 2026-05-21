"""Dataset indexing and pairing logic."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..config import DatasetConfig
from ..exceptions import DatasetError
from ..utils.filesystem import iter_files
from .manifest import AECSample, DenoiseSample, ManifestBundle


def _extract_sample_id(path: Path, separator: str) -> str:
    name = path.stem
    if separator not in name:
        raise DatasetError(f"cannot extract sample id from {path.name}")
    return name.split(separator, maxsplit=1)[-1]


def build_denoise_samples(
    noisy_root: str | Path,
    clean_root: str | Path,
    pattern: str = "*.wav",
    split: str = "train",
) -> list[DenoiseSample]:
    noisy_base = Path(noisy_root).expanduser().resolve()
    clean_base = Path(clean_root).expanduser().resolve()
    samples: list[DenoiseSample] = []
    missing: list[str] = []

    for noisy_path in iter_files(noisy_base, pattern):
        relative = noisy_path.relative_to(noisy_base)
        clean_path = clean_base / relative
        if not clean_path.exists():
            missing.append(str(relative))
            continue
        samples.append(
            DenoiseSample(
                noisy=str(noisy_path),
                clean=str(clean_path),
                split=split,
            )
        )

    if missing:
        preview = ", ".join(missing[:5])
        raise DatasetError(f"missing clean files for {len(missing)} noisy items: {preview}")
    return samples


def build_aec_samples(
    mic_root: str | Path,
    farend_root: str | Path,
    clean_root: str | Path,
    config: DatasetConfig,
    split: str = "train",
) -> list[AECSample]:
    mic_base = Path(mic_root).expanduser().resolve()
    farend_base = Path(farend_root).expanduser().resolve()
    clean_base = Path(clean_root).expanduser().resolve()
    samples: list[AECSample] = []
    missing: list[str] = []

    for mic_path in iter_files(mic_base, config.file_pattern):
        sample_id = _extract_sample_id(mic_path, config.suffix_separator)
        farend_name = f"{config.farend_prefix}{sample_id}{mic_path.suffix}"
        clean_name = f"{config.clean_prefix}{sample_id}{mic_path.suffix}"
        farend_path = farend_base / farend_name
        clean_path = clean_base / clean_name
        if not farend_path.exists() or not clean_path.exists():
            missing.append(sample_id)
            continue
        samples.append(
            AECSample(
                mic=str(mic_path),
                farend=str(farend_path),
                clean=str(clean_path),
                split=split,
                sample_id=sample_id,
            )
        )

    if missing:
        preview = ", ".join(missing[:5])
        raise DatasetError(f"missing farend or clean files for {len(missing)} ids: {preview}")
    return samples


def build_manifest_bundle(root: str | Path, config: DatasetConfig) -> ManifestBundle:
    root_path = Path(root).expanduser().resolve()
    if config.mode == "denoise":
        train = [asdict(sample) for sample in build_denoise_samples(
            config.train_noisy_root,
            config.train_clean_root,
            pattern=config.file_pattern,
            split="train",
        )]
        validation = [asdict(sample) for sample in build_denoise_samples(
            config.val_noisy_root,
            config.val_clean_root,
            pattern=config.file_pattern,
            split="validation",
        )]
        return ManifestBundle(mode="denoise", root=str(root_path), train=train, validation=validation)

    train = [asdict(sample) for sample in build_aec_samples(
        config.train_noisy_root,
        config.train_farend_root,
        config.train_clean_root,
        config,
        split="train",
    )]
    validation = [asdict(sample) for sample in build_aec_samples(
        config.val_noisy_root,
        config.val_farend_root,
        config.val_clean_root,
        config,
        split="validation",
    )]
    return ManifestBundle(mode="aec", root=str(root_path), train=train, validation=validation)
