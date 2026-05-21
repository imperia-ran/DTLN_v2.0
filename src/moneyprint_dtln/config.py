"""Typed configuration models used by training, export, and inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .exceptions import ConfigurationError

Mode = Literal["denoise", "aec"]


def _coerce_path(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser().resolve()


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000
    block_len: int = 512
    block_shift: int = 128
    clip_output: bool = True
    normalize_input: bool = False

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ConfigurationError("sample_rate must be positive")
        if self.block_len <= 0 or self.block_shift <= 0:
            raise ConfigurationError("block_len and block_shift must be positive")
        if self.block_shift > self.block_len:
            raise ConfigurationError("block_shift cannot exceed block_len")


@dataclass(slots=True)
class DatasetConfig:
    mode: Mode = "aec"
    train_noisy_root: Path | None = None
    train_clean_root: Path | None = None
    train_farend_root: Path | None = None
    val_noisy_root: Path | None = None
    val_clean_root: Path | None = None
    val_farend_root: Path | None = None
    manifest_path: Path | None = None
    file_pattern: str = "*.wav"
    mic_prefix: str = "nearend_mic_fileid_"
    farend_prefix: str = "farend_speech_fileid_"
    clean_prefix: str = "nearend_speech_fileid_"
    suffix_separator: str = "fileid_"
    allow_resample: bool = False
    chunk_seconds: float = 8.0
    batch_size: int = 16
    shuffle_buffer: int = 128
    repeat: bool = True
    drop_remainder: bool = True
    prefetch: int = 2

    def validate(self) -> None:
        if self.mode not in {"denoise", "aec"}:
            raise ConfigurationError(f"unsupported dataset mode: {self.mode}")
        if self.chunk_seconds <= 0:
            raise ConfigurationError("chunk_seconds must be positive")
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")

        for name in (
            "train_noisy_root",
            "train_clean_root",
            "train_farend_root",
            "val_noisy_root",
            "val_clean_root",
            "val_farend_root",
            "manifest_path",
        ):
            setattr(self, name, _coerce_path(getattr(self, name)))

        if self.mode == "denoise":
            required = [
                self.train_noisy_root,
                self.train_clean_root,
                self.val_noisy_root,
                self.val_clean_root,
            ]
            if any(path is None for path in required) and self.manifest_path is None:
                raise ConfigurationError(
                    "denoise mode requires train/val noisy and clean roots, or a manifest_path"
                )
        if self.mode == "aec":
            required = [
                self.train_noisy_root,
                self.train_clean_root,
                self.train_farend_root,
                self.val_noisy_root,
                self.val_clean_root,
                self.val_farend_root,
            ]
            if any(path is None for path in required) and self.manifest_path is None:
                raise ConfigurationError(
                    "aec mode requires train/val mic, farend, clean roots, or a manifest_path"
                )


@dataclass(slots=True)
class ModelConfig:
    mode: Mode = "aec"
    lstm_units: int = 128
    lstm_layers: int = 2
    encoder_size: int = 256
    activation: str = "sigmoid"
    dropout: float = 0.25
    use_log_stft_norm: bool = True
    use_safe_multiply: bool = True
    mixed_precision: bool = False
    clip_multiply: float = 10000.0

    def validate(self) -> None:
        if self.lstm_units <= 0 or self.lstm_layers <= 0 or self.encoder_size <= 0:
            raise ConfigurationError("model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigurationError("dropout must be in [0, 1)")


@dataclass(slots=True)
class TrainingConfig:
    run_name: str = "moneyprint_dtln"
    epochs: int = 200
    learning_rate: float = 1e-3
    clip_norm: float = 3.0
    patience: int = 10
    lr_patience: int = 3
    min_learning_rate: float = 1e-10
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("artifacts")
    random_seed: int = 42
    monitor: str = "val_loss"
    save_best_only: bool = True

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ConfigurationError("epochs must be positive")
        if self.learning_rate <= 0 or self.clip_norm <= 0:
            raise ConfigurationError("learning_rate and clip_norm must be positive")
        self.checkpoint_dir = _coerce_path(self.checkpoint_dir) or Path.cwd()
        self.log_dir = _coerce_path(self.log_dir) or Path.cwd()


@dataclass(slots=True)
class ExportConfig:
    export_dir: Path = Path("exports")
    weights_path: Path | None = None
    export_saved_model: bool = True
    export_tflite: bool = True
    dynamic_range_quant: bool = False
    write_keras_model: bool = False

    def validate(self) -> None:
        self.export_dir = _coerce_path(self.export_dir) or Path.cwd()
        self.weights_path = _coerce_path(self.weights_path)


@dataclass(slots=True)
class InferenceConfig:
    weights_path: Path | None = None
    model_prefix: Path | None = None
    input_root: Path | None = None
    output_root: Path | None = None
    stateful: bool = True
    tflite: bool = False
    recursive: bool = True
    mic_suffix: str = "mic.wav"
    lpb_suffix: str = "lpb.wav"

    def validate(self) -> None:
        self.weights_path = _coerce_path(self.weights_path)
        self.model_prefix = _coerce_path(self.model_prefix)
        self.input_root = _coerce_path(self.input_root)
        self.output_root = _coerce_path(self.output_root)


@dataclass(slots=True)
class ProjectConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectConfig":
        try:
            config = cls(
                audio=AudioConfig(**payload.get("audio", {})),
                dataset=DatasetConfig(**payload.get("dataset", {})),
                model=ModelConfig(**payload.get("model", {})),
                training=TrainingConfig(**payload.get("training", {})),
                export=ExportConfig(**payload.get("export", {})),
                inference=InferenceConfig(**payload.get("inference", {})),
            )
        except TypeError as exc:
            raise ConfigurationError(f"invalid config payload: {exc}") from exc
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "ProjectConfig":
        file_path = Path(path).expanduser().resolve()
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def validate(self) -> None:
        self.audio.validate()
        self.dataset.validate()
        self.model.validate()
        self.training.validate()
        self.export.validate()
        self.inference.validate()
        if self.model.mode != self.dataset.mode:
            raise ConfigurationError("model.mode must match dataset.mode")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
