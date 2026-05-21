"""TensorFlow dataset builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Iterator

import numpy as np
from wavinfo import WavInfoReader

from ..config import AudioConfig, DatasetConfig
from ..exceptions import DatasetError
from ..tf_compat import require_tensorflow
from ..utils.audio import ensure_same_length, read_mono_audio, sliding_chunks
from ..utils.filesystem import relative_to_root
from .manifest import ManifestBundle, load_manifest
from .pairs import build_aec_samples, build_denoise_samples


@dataclass(slots=True)
class SampleCount:
    items: int
    chunks: int


class DatasetBuilder:
    """Builds train and validation tf.data pipelines from roots or a manifest."""

    def __init__(self, audio: AudioConfig, dataset: DatasetConfig, seed: int = 42) -> None:
        self.audio = audio
        self.dataset = dataset
        self.random = Random(seed)

    def _chunk_size(self) -> int:
        chunks = int(self.audio.sample_rate * self.dataset.chunk_seconds)
        return chunks - (chunks % self.audio.block_shift)

    def _load_manifest(self) -> ManifestBundle | None:
        if self.dataset.manifest_path is None:
            return None
        return load_manifest(self.dataset.manifest_path)

    def count_split(self, split: str) -> SampleCount:
        samples = self._resolve_split(split)
        chunk_size = self._chunk_size()
        total_chunks = 0
        for sample in samples:
            if self.dataset.mode == "denoise":
                info = WavInfoReader(str(Path(sample["noisy"])))
            else:
                info = WavInfoReader(str(Path(sample["mic"])))
            total_chunks += int(info.data.frame_count // chunk_size)
        return SampleCount(items=len(samples), chunks=total_chunks)

    def _resolve_split(self, split: str) -> list[dict[str, str]]:
        manifest = self._load_manifest()
        if manifest is not None:
            return manifest.train if split == "train" else manifest.validation

        if self.dataset.mode == "denoise":
            source = build_denoise_samples(
                self.dataset.train_noisy_root if split == "train" else self.dataset.val_noisy_root,
                self.dataset.train_clean_root if split == "train" else self.dataset.val_clean_root,
                pattern=self.dataset.file_pattern,
                split=split,
            )
            return [asdict(sample) for sample in source]

        source = build_aec_samples(
            self.dataset.train_noisy_root if split == "train" else self.dataset.val_noisy_root,
            self.dataset.train_farend_root if split == "train" else self.dataset.val_farend_root,
            self.dataset.train_clean_root if split == "train" else self.dataset.val_clean_root,
            self.dataset,
            split=split,
        )
        return [asdict(sample) for sample in source]

    def _iter_denoise(self, samples: list[dict[str, str]]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        chunk_size = self._chunk_size()
        if self.dataset.repeat and samples:
            self.random.shuffle(samples)
        for sample in samples:
            noisy = read_mono_audio(sample["noisy"], expected_sample_rate=self.audio.sample_rate).data
            clean = read_mono_audio(sample["clean"], expected_sample_rate=self.audio.sample_rate).data
            noisy, clean = ensure_same_length(noisy, clean)
            for noisy_chunk, clean_chunk in zip(
                sliding_chunks(noisy, chunk_size),
                sliding_chunks(clean, chunk_size),
                strict=False,
            ):
                yield noisy_chunk.astype(np.float32), clean_chunk.astype(np.float32)

    def _iter_aec(self, samples: list[dict[str, str]]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        chunk_size = self._chunk_size()
        if self.dataset.repeat and samples:
            self.random.shuffle(samples)
        for sample in samples:
            mic = read_mono_audio(sample["mic"], expected_sample_rate=self.audio.sample_rate).data
            farend = read_mono_audio(sample["farend"], expected_sample_rate=self.audio.sample_rate).data
            clean = read_mono_audio(sample["clean"], expected_sample_rate=self.audio.sample_rate).data
            mic, farend, clean = ensure_same_length(mic, farend, clean)
            for mic_chunk, farend_chunk, clean_chunk in zip(
                sliding_chunks(mic, chunk_size),
                sliding_chunks(farend, chunk_size),
                sliding_chunks(clean, chunk_size),
                strict=False,
            ):
                features = np.concatenate((mic_chunk, farend_chunk)).astype(np.float32)
                yield features, clean_chunk.astype(np.float32)

    def iter_split(self, split: str) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        samples = self._resolve_split(split)
        if not samples:
            raise DatasetError(f"no samples resolved for split={split}")
        if self.dataset.mode == "denoise":
            yield from self._iter_denoise(samples)
            return
        yield from self._iter_aec(samples)

    def build_tf_dataset(self, split: str):
        tf = require_tensorflow()
        chunk_size = self._chunk_size()
        feature_size = chunk_size if self.dataset.mode == "denoise" else chunk_size * 2
        dataset = tf.data.Dataset.from_generator(
            lambda: self.iter_split(split),
            output_signature=(
                tf.TensorSpec(shape=(feature_size,), dtype=tf.float32),
                tf.TensorSpec(shape=(chunk_size,), dtype=tf.float32),
            ),
        )
        if split == "train":
            dataset = dataset.shuffle(self.dataset.shuffle_buffer)
        dataset = dataset.batch(self.dataset.batch_size, drop_remainder=self.dataset.drop_remainder)
        if self.dataset.repeat:
            dataset = dataset.repeat()
        return dataset.prefetch(self.dataset.prefetch)
