"""Export helpers for SavedModel and TFLite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .models.aec import DTLNAECModel
from .models.noise import DTLNNoiseModel
from .tf_compat import require_tensorflow
from .utils.filesystem import ensure_directory


class ExportService:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.tf = require_tensorflow()

    def _stateful_model(self) -> Any:
        if self.config.model.mode == "denoise":
            builder = DTLNNoiseModel(self.config.audio, self.config.model)
            return builder.build_stateful_model()
        builder = DTLNAECModel(self.config.audio, self.config.model)
        return builder.build_stateful_model()

    def export(self) -> dict[str, Path]:
        target_dir = ensure_directory(self.config.export.export_dir)
        outputs: dict[str, Path] = {}
        model = self._stateful_model()
        model.load_weights(str(self.config.export.weights_path))

        if self.config.export.export_saved_model:
            saved_model_dir = target_dir / "saved_model"
            self.tf.saved_model.save(model, str(saved_model_dir))
            outputs["saved_model"] = saved_model_dir

        if self.config.export.write_keras_model:
            keras_path = target_dir / "model.keras"
            self.tf.keras.saving.save_model(model, str(keras_path))
            outputs["keras_model"] = keras_path

        if self.config.export.export_tflite:
            if self.config.model.mode == "denoise":
                outputs.update(self._export_noise_tflite(model, target_dir))
            else:
                outputs.update(self._export_aec_tflite(model, target_dir))
        return outputs

    def _convert_tflite(self, keras_model: Any, quantize: bool) -> bytes:
        converter = self.tf.lite.TFLiteConverter.from_keras_model(keras_model)
        if quantize:
            converter.optimizations = [self.tf.lite.Optimize.DEFAULT]
        return converter.convert()

    def _export_noise_tflite(self, model: Any, target_dir: Path) -> dict[str, Path]:
        outputs: dict[str, Path] = {}
        binary = self._convert_tflite(model, self.config.export.dynamic_range_quant)
        tflite_path = target_dir / "dtln_noise.tflite"
        tflite_path.write_bytes(binary)
        outputs["tflite"] = tflite_path
        return outputs

    def _export_aec_tflite(self, stateful_model: Any, target_dir: Path) -> dict[str, Path]:
        builder = DTLNAECModel(self.config.audio, self.config.model)
        stage_one, stage_two = builder.build_tflite_pair()
        weights = stateful_model.get_weights()

        stage_one_weight_count = len(stage_one.get_weights())
        stage_one.set_weights(weights[:stage_one_weight_count])
        stage_two.set_weights(weights[stage_one_weight_count:])

        outputs: dict[str, Path] = {}
        for label, keras_model in (("tflite_stage_1", stage_one), ("tflite_stage_2", stage_two)):
            suffix = "_1.tflite" if label.endswith("1") else "_2.tflite"
            binary = self._convert_tflite(keras_model, self.config.export.dynamic_range_quant)
            file_path = target_dir / f"dtln_aec{suffix}"
            file_path.write_bytes(binary)
            outputs[label] = file_path
        return outputs
