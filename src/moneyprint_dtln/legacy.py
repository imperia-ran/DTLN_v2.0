"""Compatibility adapters for the previous repository layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AudioConfig, DatasetConfig, ExportConfig, ModelConfig, ProjectConfig, TrainingConfig
from .exporters import ExportService
from .models.aec import DTLNAECModel
from .models.noise import DTLNNoiseModel
from .training import Trainer


class LegacyDTLNFacade:
    """Expose a familiar surface for older scripts while delegating to the new stack."""

    def __init__(self, mode: str = "aec") -> None:
        self.audio = AudioConfig(block_shift=256 if mode == "aec" else 128)
        self.model_config = ModelConfig(mode=mode, lstm_units=64 if mode == "aec" else 128)
        self.dataset = DatasetConfig(mode=mode)
        self.training = TrainingConfig(run_name="legacy_dtln")
        self.export = ExportConfig()
        self.model: Any = None

    @property
    def fs(self) -> int:
        return self.audio.sample_rate

    @property
    def blockLen(self) -> int:
        return self.audio.block_len

    @property
    def block_shift(self) -> int:
        return self.audio.block_shift

    @property
    def encoder_size(self) -> int:
        return self.model_config.encoder_size

    @property
    def numUnits(self) -> int:
        return self.model_config.lstm_units

    @property
    def numLayer(self) -> int:
        return self.model_config.lstm_layers

    @property
    def dropout(self) -> float:
        return self.model_config.dropout

    @dropout.setter
    def dropout(self, value: float) -> None:
        self.model_config.dropout = value

    @property
    def lr(self) -> float:
        return self.training.learning_rate

    @lr.setter
    def lr(self, value: float) -> None:
        self.training.learning_rate = value

    @property
    def batchsize(self) -> int:
        return self.dataset.batch_size

    @batchsize.setter
    def batchsize(self, value: int) -> None:
        self.dataset.batch_size = value

    @property
    def max_epochs(self) -> int:
        return self.training.epochs

    @max_epochs.setter
    def max_epochs(self, value: int) -> None:
        self.training.epochs = value

    def _builder(self) -> Any:
        if self.model_config.mode == "denoise":
            return DTLNNoiseModel(self.audio, self.model_config)
        return DTLNAECModel(self.audio, self.model_config)

    def build_DTLN_model(self, norm_stft: bool = True) -> Any:
        self.model_config.use_log_stft_norm = norm_stft
        self.model = self._builder().build_training_model()
        return self.model

    def build_DTLN_model_stateful(self, norm_stft: bool = True) -> Any:
        self.model_config.use_log_stft_norm = norm_stft
        self.model = self._builder().build_stateful_model()
        return self.model

    def compile_model(self) -> Any:
        builder = self._builder()
        if self.model is None:
            self.build_DTLN_model()
        if self.model_config.mode == "denoise":
            self.model = builder.compile(self.model)
        else:
            self.model = builder.compile(self.model, self.training.learning_rate, self.training.clip_norm)
        return self.model

    def create_saved_model(self, weights_file: str, target_name: str) -> dict[str, Path]:
        self.export.weights_path = Path(weights_file)
        self.export.export_dir = Path(target_name)
        self.export.export_tflite = False
        self.export.export_saved_model = True
        self.export.write_keras_model = False
        config = ProjectConfig(
            audio=self.audio,
            dataset=self.dataset,
            model=self.model_config,
            training=self.training,
            export=self.export,
        )
        return ExportService(config).export()

    def create_tf_lite_model(
        self,
        weights_file: str,
        target_name: str,
        use_dynamic_range_quant: bool = False,
    ) -> dict[str, Path]:
        self.export.weights_path = Path(weights_file)
        self.export.export_dir = Path(target_name).parent
        self.export.export_tflite = True
        self.export.export_saved_model = False
        self.export.dynamic_range_quant = use_dynamic_range_quant
        config = ProjectConfig(
            audio=self.audio,
            dataset=self.dataset,
            model=self.model_config,
            training=self.training,
            export=self.export,
        )
        return ExportService(config).export()

    def train_model(
        self,
        run_name: str,
        path_to_train_mic: str,
        path_to_train_lpb: str | None,
        path_to_train_speech: str,
        path_to_val_mic: str,
        path_to_val_lpb: str | None,
        path_to_val_speech: str,
    ) -> Path:
        self.training.run_name = run_name
        self.dataset.train_noisy_root = Path(path_to_train_mic)
        self.dataset.train_clean_root = Path(path_to_train_speech)
        self.dataset.val_noisy_root = Path(path_to_val_mic)
        self.dataset.val_clean_root = Path(path_to_val_speech)
        if self.model_config.mode == "aec":
            self.dataset.train_farend_root = Path(path_to_train_lpb or "")
            self.dataset.val_farend_root = Path(path_to_val_lpb or "")
        config = ProjectConfig(
            audio=self.audio,
            dataset=self.dataset,
            model=self.model_config,
            training=self.training,
            export=self.export,
        )
        artifacts = Trainer(config).train()
        return artifacts.checkpoint_path


class DTLN_model(LegacyDTLNFacade):
    def __init__(self) -> None:
        super().__init__(mode="denoise")


class DTLN_aecmodel(LegacyDTLNFacade):
    def __init__(self) -> None:
        super().__init__(mode="aec")
