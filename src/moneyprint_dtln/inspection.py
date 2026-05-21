"""Dataset and model inspection routines for operators and debugging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import AudioConfig, ModelConfig, ProjectConfig
from .data.dataset import DatasetBuilder
from .preprocess import inspect_dataset_config, summarize_durations
from .tf_compat import require_tensorflow


@dataclass(slots=True)
class ModelSummary:
    name: str
    parameters: int
    output_shapes: list[str] = field(default_factory=list)
    input_shapes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DatasetSummary:
    mode: str
    train_items: int
    train_chunks: int
    validation_items: int
    validation_chunks: int
    duration_stats: dict[str, float] = field(default_factory=dict)
    issue_count: int = 0


def summarize_model(model: Any) -> ModelSummary:
    inputs = []
    outputs = []
    for item in model.inputs:
        inputs.append(str(item.shape))
    for item in model.outputs:
        outputs.append(str(item.shape))
    return ModelSummary(
        name=model.name,
        parameters=int(model.count_params()),
        input_shapes=inputs,
        output_shapes=outputs,
    )


def inspect_model_builder(audio: AudioConfig, model_config: ModelConfig) -> ModelSummary:
    tf = require_tensorflow()
    from .models.aec import DTLNAECModel
    from .models.noise import DTLNNoiseModel

    if model_config.mode == "denoise":
        builder = DTLNNoiseModel(audio, model_config)
        model = builder.build_training_model()
    else:
        builder = DTLNAECModel(audio, model_config)
        model = builder.build_training_model()

    summary = summarize_model(model)
    tf.keras.backend.clear_session()
    return summary


def inspect_dataset(project: ProjectConfig) -> DatasetSummary:
    dataset_builder = DatasetBuilder(project.audio, project.dataset, seed=project.training.random_seed)
    train = dataset_builder.count_split("train")
    validation = dataset_builder.count_split("validation")
    validation_report = inspect_dataset_config(project.dataset, project.audio)
    return DatasetSummary(
        mode=project.dataset.mode,
        train_items=train.items,
        train_chunks=train.chunks,
        validation_items=validation.items,
        validation_chunks=validation.chunks,
        duration_stats=summarize_durations(validation_report.metadata),
        issue_count=len(validation_report.issues),
    )


def format_dataset_summary(summary: DatasetSummary) -> str:
    lines = [
        f"mode: {summary.mode}",
        f"train_items: {summary.train_items}",
        f"train_chunks: {summary.train_chunks}",
        f"validation_items: {summary.validation_items}",
        f"validation_chunks: {summary.validation_chunks}",
        f"issue_count: {summary.issue_count}",
    ]
    for key, value in summary.duration_stats.items():
        lines.append(f"{key}: {value:.2f}")
    return "\n".join(lines)


def format_model_summary(summary: ModelSummary) -> str:
    lines = [
        f"name: {summary.name}",
        f"parameters: {summary.parameters}",
        "inputs:",
    ]
    lines.extend(summary.input_shapes)
    lines.append("outputs:")
    lines.extend(summary.output_shapes)
    return "\n".join(lines)


def save_summary(text: str, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.write_text(text, encoding="utf-8")
    return target


def dry_run_dataset_batch(project: ProjectConfig) -> dict[str, Any]:
    tf = require_tensorflow()
    builder = DatasetBuilder(project.audio, project.dataset, seed=project.training.random_seed)
    dataset = builder.build_tf_dataset("train")
    batch = next(iter(dataset.take(1)))
    features, targets = batch
    payload = {
        "feature_shape": tuple(int(x) for x in features.shape),
        "target_shape": tuple(int(x) for x in targets.shape),
        "feature_mean": float(tf.reduce_mean(features).numpy()),
        "target_mean": float(tf.reduce_mean(targets).numpy()),
        "feature_std": float(tf.math.reduce_std(features).numpy()),
        "target_std": float(tf.math.reduce_std(targets).numpy()),
    }
    tf.keras.backend.clear_session()
    return payload


def compare_signals(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float32)
    estimate = estimate.astype(np.float32)
    error = reference - estimate
    power = float(np.mean(reference**2)) if reference.size else 0.0
    error_power = float(np.mean(error**2)) if error.size else 0.0
    return {
        "reference_power": power,
        "error_power": error_power,
        "peak_reference": float(np.max(np.abs(reference))) if reference.size else 0.0,
        "peak_estimate": float(np.max(np.abs(estimate))) if estimate.size else 0.0,
    }
