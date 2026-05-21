from __future__ import annotations

import json
from pathlib import Path

import pytest

from moneyprint_dtln.config import AudioConfig, DatasetConfig, ModelConfig, ProjectConfig
from moneyprint_dtln.exceptions import ConfigurationError


def test_audio_config_rejects_invalid_block_sizes() -> None:
    config = AudioConfig(sample_rate=16000, block_len=128, block_shift=256)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_dataset_config_requires_aec_roots_or_manifest() -> None:
    config = DatasetConfig(mode="aec")
    with pytest.raises(ConfigurationError):
        config.validate()


def test_dataset_config_requires_denoise_roots_or_manifest() -> None:
    config = DatasetConfig(mode="denoise")
    with pytest.raises(ConfigurationError):
        config.validate()


def test_model_config_validates_dropout() -> None:
    config = ModelConfig(mode="aec", dropout=1.2)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_project_config_from_dict_roundtrip(tmp_path: Path) -> None:
    payload = {
        "audio": {
            "sample_rate": 16000,
            "block_len": 512,
            "block_shift": 256,
        },
        "dataset": {
            "mode": "aec",
            "train_noisy_root": str(tmp_path / "train_mic"),
            "train_clean_root": str(tmp_path / "train_clean"),
            "train_farend_root": str(tmp_path / "train_farend"),
            "val_noisy_root": str(tmp_path / "val_mic"),
            "val_clean_root": str(tmp_path / "val_clean"),
            "val_farend_root": str(tmp_path / "val_farend"),
        },
        "model": {
            "mode": "aec",
            "lstm_units": 64,
            "lstm_layers": 2,
            "encoder_size": 192,
        },
        "training": {
            "run_name": "unit_test",
            "epochs": 2,
            "learning_rate": 0.001,
            "clip_norm": 3.0,
            "patience": 1,
            "lr_patience": 1,
            "min_learning_rate": 1e-10,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "artifacts"),
            "random_seed": 42,
            "monitor": "val_loss",
            "save_best_only": True,
        },
        "export": {
            "export_dir": str(tmp_path / "exports"),
            "weights_path": None,
            "export_saved_model": True,
            "export_tflite": False,
            "dynamic_range_quant": False,
            "write_keras_model": False,
        },
        "inference": {
            "weights_path": None,
            "model_prefix": None,
            "input_root": None,
            "output_root": None,
            "stateful": True,
            "tflite": False,
            "recursive": True,
            "mic_suffix": "mic.wav",
            "lpb_suffix": "lpb.wav",
        },
    }
    config = ProjectConfig.from_dict(payload)
    roundtrip = config.to_dict()
    assert roundtrip["dataset"]["mode"] == "aec"
    assert roundtrip["model"]["encoder_size"] == 192


def test_project_config_from_json(tmp_path: Path) -> None:
    payload = {
        "audio": {"sample_rate": 16000, "block_len": 512, "block_shift": 128},
        "dataset": {
            "mode": "denoise",
            "train_noisy_root": str(tmp_path / "train_noisy"),
            "train_clean_root": str(tmp_path / "train_clean"),
            "val_noisy_root": str(tmp_path / "val_noisy"),
            "val_clean_root": str(tmp_path / "val_clean"),
        },
        "model": {"mode": "denoise", "lstm_units": 128, "lstm_layers": 2, "encoder_size": 256},
        "training": {
            "run_name": "unit_json",
            "epochs": 1,
            "learning_rate": 0.001,
            "clip_norm": 3.0,
            "patience": 1,
            "lr_patience": 1,
            "min_learning_rate": 1e-10,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "log_dir": str(tmp_path / "log"),
            "random_seed": 42,
            "monitor": "val_loss",
            "save_best_only": True,
        },
        "export": {
            "export_dir": str(tmp_path / "exports"),
            "weights_path": None,
            "export_saved_model": True,
            "export_tflite": False,
            "dynamic_range_quant": False,
            "write_keras_model": False,
        },
        "inference": {
            "weights_path": None,
            "model_prefix": None,
            "input_root": None,
            "output_root": None,
            "stateful": True,
            "tflite": False,
            "recursive": True,
            "mic_suffix": "mic.wav",
            "lpb_suffix": "lpb.wav",
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = ProjectConfig.from_json(path)
    assert config.dataset.mode == "denoise"
    assert config.audio.block_shift == 128


def test_project_config_rejects_mismatched_modes(tmp_path: Path) -> None:
    payload = {
        "audio": {"sample_rate": 16000, "block_len": 512, "block_shift": 128},
        "dataset": {
            "mode": "denoise",
            "train_noisy_root": str(tmp_path / "train_noisy"),
            "train_clean_root": str(tmp_path / "train_clean"),
            "val_noisy_root": str(tmp_path / "val_noisy"),
            "val_clean_root": str(tmp_path / "val_clean"),
        },
        "model": {"mode": "aec", "lstm_units": 64, "lstm_layers": 2, "encoder_size": 192},
        "training": {
            "run_name": "bad",
            "epochs": 1,
            "learning_rate": 0.001,
            "clip_norm": 3.0,
            "patience": 1,
            "lr_patience": 1,
            "min_learning_rate": 1e-10,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "log_dir": str(tmp_path / "log"),
            "random_seed": 42,
            "monitor": "val_loss",
            "save_best_only": True,
        },
        "export": {"export_dir": str(tmp_path / "exports")},
        "inference": {"stateful": True, "tflite": False, "recursive": True, "mic_suffix": "mic.wav", "lpb_suffix": "lpb.wav"},
    }
    with pytest.raises(ConfigurationError):
        ProjectConfig.from_dict(payload)
