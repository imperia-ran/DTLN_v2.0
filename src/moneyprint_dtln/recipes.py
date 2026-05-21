"""Programmatic config presets for common workflows."""

from __future__ import annotations

from pathlib import Path

from .config import AudioConfig, DatasetConfig, ExportConfig, InferenceConfig, ModelConfig, ProjectConfig, TrainingConfig


def aec_training_recipe(dataset_root: str | Path, run_name: str = "dtln_aec_recipe") -> ProjectConfig:
    root = Path(dataset_root).expanduser().resolve()
    return ProjectConfig(
        audio=AudioConfig(sample_rate=16000, block_len=512, block_shift=256),
        dataset=DatasetConfig(
            mode="aec",
            train_noisy_root=root / "train" / "nearend_mic_signal",
            train_clean_root=root / "train" / "nearend_speech",
            train_farend_root=root / "train" / "farend_speech",
            val_noisy_root=root / "val" / "nearend_mic_signal",
            val_clean_root=root / "val" / "nearend_speech",
            val_farend_root=root / "val" / "farend_speech",
            chunk_seconds=8.0,
            batch_size=16,
        ),
        model=ModelConfig(
            mode="aec",
            lstm_units=64,
            lstm_layers=2,
            encoder_size=192,
            use_log_stft_norm=True,
            use_safe_multiply=True,
        ),
        training=TrainingConfig(run_name=run_name),
        export=ExportConfig(),
        inference=InferenceConfig(),
    )


def denoise_training_recipe(dataset_root: str | Path, run_name: str = "dtln_denoise_recipe") -> ProjectConfig:
    root = Path(dataset_root).expanduser().resolve()
    return ProjectConfig(
        audio=AudioConfig(sample_rate=16000, block_len=512, block_shift=128),
        dataset=DatasetConfig(
            mode="denoise",
            train_noisy_root=root / "train" / "noisy",
            train_clean_root=root / "train" / "clean",
            val_noisy_root=root / "val" / "noisy",
            val_clean_root=root / "val" / "clean",
            chunk_seconds=15.0,
            batch_size=32,
        ),
        model=ModelConfig(
            mode="denoise",
            lstm_units=128,
            lstm_layers=2,
            encoder_size=256,
            use_log_stft_norm=True,
            use_safe_multiply=False,
        ),
        training=TrainingConfig(run_name=run_name),
        export=ExportConfig(),
        inference=InferenceConfig(),
    )


def export_recipe(weights_path: str | Path, export_dir: str | Path, mode: str = "aec") -> ProjectConfig:
    config = aec_training_recipe(Path(".") if mode == "aec" else Path("."), run_name="export") if mode == "aec" else denoise_training_recipe(Path("."), run_name="export")
    config.export.weights_path = Path(weights_path).expanduser().resolve()
    config.export.export_dir = Path(export_dir).expanduser().resolve()
    return config


def tflite_inference_recipe(model_prefix: str | Path, input_root: str | Path, output_root: str | Path) -> ProjectConfig:
    config = aec_training_recipe(Path("."), run_name="tflite_infer")
    config.dataset.train_noisy_root = Path("/unused")
    config.dataset.train_clean_root = Path("/unused")
    config.dataset.train_farend_root = Path("/unused")
    config.dataset.val_noisy_root = Path("/unused")
    config.dataset.val_clean_root = Path("/unused")
    config.dataset.val_farend_root = Path("/unused")
    config.inference.model_prefix = Path(model_prefix).expanduser().resolve()
    config.inference.input_root = Path(input_root).expanduser().resolve()
    config.inference.output_root = Path(output_root).expanduser().resolve()
    config.inference.tflite = True
    return config
