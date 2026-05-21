from __future__ import annotations

from pathlib import Path

from moneyprint_dtln.recipes import aec_training_recipe, denoise_training_recipe, export_recipe, tflite_inference_recipe


def test_aec_training_recipe_populates_expected_roots(tmp_path: Path) -> None:
    config = aec_training_recipe(tmp_path, run_name="aec_recipe")
    assert config.dataset.mode == "aec"
    assert str(config.dataset.train_noisy_root).endswith("train\\nearend_mic_signal") or str(config.dataset.train_noisy_root).endswith("train/nearend_mic_signal")
    assert config.training.run_name == "aec_recipe"


def test_denoise_training_recipe_populates_expected_roots(tmp_path: Path) -> None:
    config = denoise_training_recipe(tmp_path, run_name="denoise_recipe")
    assert config.dataset.mode == "denoise"
    assert str(config.dataset.train_clean_root).endswith("train\\clean") or str(config.dataset.train_clean_root).endswith("train/clean")
    assert config.training.run_name == "denoise_recipe"


def test_export_recipe_sets_weights_and_export_dir(tmp_path: Path) -> None:
    weights = tmp_path / "model.weights.h5"
    export_dir = tmp_path / "exports"
    config = export_recipe(weights, export_dir, mode="aec")
    assert config.export.weights_path == weights.resolve()
    assert config.export.export_dir == export_dir.resolve()


def test_tflite_inference_recipe_sets_runtime_paths(tmp_path: Path) -> None:
    prefix = tmp_path / "model_prefix"
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    config = tflite_inference_recipe(prefix, input_root, output_root)
    assert config.inference.tflite is True
    assert config.inference.model_prefix == prefix.resolve()
    assert config.inference.input_root == input_root.resolve()
    assert config.inference.output_root == output_root.resolve()
