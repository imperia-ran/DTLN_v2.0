"""Training orchestration for the rewritten project."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .data.dataset import DatasetBuilder
from .models.aec import DTLNAECModel
from .models.noise import DTLNNoiseModel
from .tf_compat import require_tensorflow
from .utils.filesystem import ensure_directory


@dataclass(slots=True)
class TrainingArtifacts:
    checkpoint_path: Path
    csv_log_path: Path
    run_dir: Path


class Trainer:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.tf = require_tensorflow()

    def _set_runtime(self) -> None:
        os.environ["PYTHONHASHSEED"] = str(self.config.training.random_seed)
        random.seed(self.config.training.random_seed)
        np.random.seed(self.config.training.random_seed)
        self.tf.random.set_seed(self.config.training.random_seed)
        devices = self.tf.config.experimental.list_physical_devices("GPU")
        for device in devices:
            self.tf.config.experimental.set_memory_growth(device, enable=True)

    def _build_model(self) -> Any:
        if self.config.model.mode == "denoise":
            builder = DTLNNoiseModel(self.config.audio, self.config.model)
            model = builder.build_training_model()
            return builder.compile(model)
        builder = DTLNAECModel(self.config.audio, self.config.model)
        model = builder.build_training_model()
        return builder.compile(model, self.config.training.learning_rate, self.config.training.clip_norm)

    def _callbacks(self, run_dir: Path) -> tuple[list[Any], TrainingArtifacts]:
        callbacks = self.tf.keras.callbacks
        run_dir = ensure_directory(run_dir)
        csv_log_path = run_dir / f"training_{self.config.training.run_name}.log"
        checkpoint_path = run_dir / f"{self.config.training.run_name}.weights.h5"
        callback_list = [
            callbacks.CSVLogger(str(csv_log_path)),
            callbacks.ReduceLROnPlateau(
                monitor=self.config.training.monitor,
                factor=0.5,
                patience=self.config.training.lr_patience,
                min_lr=self.config.training.min_learning_rate,
                cooldown=1,
            ),
            callbacks.EarlyStopping(
                monitor=self.config.training.monitor,
                min_delta=0.0,
                patience=self.config.training.patience,
                verbose=1,
            ),
            callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=self.config.training.monitor,
                verbose=1,
                save_best_only=self.config.training.save_best_only,
                save_weights_only=True,
                mode="auto",
                save_freq="epoch",
            ),
        ]
        artifacts = TrainingArtifacts(
            checkpoint_path=checkpoint_path,
            csv_log_path=csv_log_path,
            run_dir=run_dir,
        )
        return callback_list, artifacts

    def train(self) -> TrainingArtifacts:
        self._set_runtime()
        run_dir = ensure_directory(self.config.training.checkpoint_dir / self.config.training.run_name)
        dataset_builder = DatasetBuilder(
            audio=self.config.audio,
            dataset=self.config.dataset,
            seed=self.config.training.random_seed,
        )
        train_dataset = dataset_builder.build_tf_dataset("train")
        val_dataset = dataset_builder.build_tf_dataset("validation")
        train_count = dataset_builder.count_split("train")
        val_count = dataset_builder.count_split("validation")

        steps_per_epoch = max(1, train_count.chunks // self.config.dataset.batch_size)
        validation_steps = max(1, val_count.chunks // self.config.dataset.batch_size)
        model = self._build_model()
        callback_list, artifacts = self._callbacks(run_dir)

        model.fit(
            x=train_dataset,
            batch_size=None,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.training.epochs,
            verbose=1,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callback_list,
        )
        self.tf.keras.backend.clear_session()
        return artifacts
