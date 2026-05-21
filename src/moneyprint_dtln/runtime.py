"""Offline and streaming inference for Keras and TFLite models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from .config import AudioConfig, InferenceConfig, ModelConfig
from .models.aec import DTLNAECModel
from .models.noise import DTLNNoiseModel
from .signal import SignalTransform
from .tf_compat import require_tensorflow
from .utils.audio import clip_audio, ensure_same_length, pad_signal, read_mono_audio
from .utils.filesystem import ensure_directory


@dataclass(slots=True)
class RuntimeState:
    stage_one: np.ndarray
    stage_two: np.ndarray


class OfflineEnhancer:
    def __init__(self, audio: AudioConfig, model: ModelConfig, inference: InferenceConfig) -> None:
        self.audio = audio
        self.model = model
        self.inference = inference
        self.signal = SignalTransform(audio.block_len, audio.block_shift)

    def _padding(self) -> int:
        return self.audio.block_len - self.audio.block_shift

    def _load_keras_model(self) -> Any:
        tf = require_tensorflow()
        if self.model.mode == "denoise":
            builder = DTLNNoiseModel(self.audio, self.model)
            keras_model = builder.build_stateful_model()
        else:
            builder = DTLNAECModel(self.audio, self.model)
            keras_model = builder.build_stateful_model()
        keras_model.load_weights(str(self.inference.weights_path))
        return keras_model, tf

    def process_file(self, input_path: str | Path, output_path: str | Path, farend_path: str | Path | None = None) -> Path:
        output = ensure_directory(Path(output_path).expanduser().resolve().parent) / Path(output_path).name
        noisy = read_mono_audio(input_path, expected_sample_rate=self.audio.sample_rate).data

        if self.model.mode == "denoise":
            enhanced = self._process_denoise(noisy)
        else:
            inferred_farend = farend_path
            if inferred_farend is None:
                inferred_farend = str(input_path).replace(self.inference.mic_suffix, self.inference.lpb_suffix)
            farend = read_mono_audio(inferred_farend, expected_sample_rate=self.audio.sample_rate).data
            enhanced = self._process_aec(noisy, farend)

        sf.write(str(output), clip_audio(enhanced), self.audio.sample_rate)
        return output

    def _process_denoise(self, noisy: np.ndarray) -> np.ndarray:
        model, _ = self._load_keras_model()
        padding = self._padding()
        padded = pad_signal(noisy, padding, padding)
        output = np.zeros(len(padded), dtype=np.float32)
        buffer_in = np.zeros(self.audio.block_len, dtype=np.float32)
        buffer_out = np.zeros(self.audio.block_len, dtype=np.float32)
        blocks = (len(padded) - padding) // self.audio.block_shift

        for idx in range(blocks):
            buffer_in[:-self.audio.block_shift] = buffer_in[self.audio.block_shift :]
            buffer_in[-self.audio.block_shift :] = padded[
                idx * self.audio.block_shift : (idx * self.audio.block_shift) + self.audio.block_shift
            ]
            frame = np.reshape(buffer_in, (1, self.audio.block_len)).astype(np.float32)
            block = np.squeeze(model(frame, training=False).numpy())
            buffer_out[:-self.audio.block_shift] = buffer_out[self.audio.block_shift :]
            buffer_out[-self.audio.block_shift :] = 0.0
            buffer_out += block
            output[idx * self.audio.block_shift : (idx * self.audio.block_shift) + self.audio.block_shift] = buffer_out[
                : self.audio.block_shift
            ]

        return output[padding : padding + len(noisy)]

    def _process_aec(self, mic: np.ndarray, farend: np.ndarray) -> np.ndarray:
        model, _ = self._load_keras_model()
        mic, farend = ensure_same_length(mic, farend)
        original_length = len(mic)
        padding = self._padding()
        mic = pad_signal(mic, padding, padding)
        farend = pad_signal(farend, padding, padding)

        output = np.zeros(len(mic), dtype=np.float32)
        mic_buffer = np.zeros(self.audio.block_len, dtype=np.float32)
        farend_buffer = np.zeros(self.audio.block_len, dtype=np.float32)
        out_buffer = np.zeros(self.audio.block_len, dtype=np.float32)
        blocks = (len(mic) - padding) // self.audio.block_shift

        for idx in range(blocks):
            mic_buffer[:-self.audio.block_shift] = mic_buffer[self.audio.block_shift :]
            mic_buffer[-self.audio.block_shift :] = mic[
                idx * self.audio.block_shift : (idx * self.audio.block_shift) + self.audio.block_shift
            ]
            farend_buffer[:-self.audio.block_shift] = farend_buffer[self.audio.block_shift :]
            farend_buffer[-self.audio.block_shift :] = farend[
                idx * self.audio.block_shift : (idx * self.audio.block_shift) + self.audio.block_shift
            ]
            mic_frame = np.reshape(mic_buffer, (1, self.audio.block_len)).astype(np.float32)
            farend_frame = np.reshape(farend_buffer, (1, self.audio.block_len)).astype(np.float32)
            block = np.squeeze(model([mic_frame, farend_frame], training=False).numpy())
            out_buffer[:-self.audio.block_shift] = out_buffer[self.audio.block_shift :]
            out_buffer[-self.audio.block_shift :] = 0.0
            out_buffer += block
            output[idx * self.audio.block_shift : (idx * self.audio.block_shift) + self.audio.block_shift] = out_buffer[
                : self.audio.block_shift
            ]

        return output[padding : padding + original_length]


class TFLiteAECEnhancer:
    def __init__(self, audio: AudioConfig, model_prefix: str | Path) -> None:
        tf = require_tensorflow()
        self.audio = audio
        self.signal = SignalTransform(audio.block_len, audio.block_shift)
        prefix = Path(model_prefix).expanduser().resolve()
        self.stage_one = tf.lite.Interpreter(model_path=str(prefix.parent / f"{prefix.name}_1.tflite"))
        self.stage_two = tf.lite.Interpreter(model_path=str(prefix.parent / f"{prefix.name}_2.tflite"))
        self.stage_one.allocate_tensors()
        self.stage_two.allocate_tensors()
        self.input_one = self.stage_one.get_input_details()
        self.output_one = self.stage_one.get_output_details()
        self.input_two = self.stage_two.get_input_details()
        self.output_two = self.stage_two.get_output_details()

    def _initial_state(self, details: list[dict[str, Any]], index: int) -> np.ndarray:
        return np.zeros(details[index]["shape"], dtype=np.float32)

    def process(self, mic: np.ndarray, farend: np.ndarray) -> np.ndarray:
        mic, farend = ensure_same_length(mic, farend)
        padding = self.audio.block_len - self.audio.block_shift
        mic = pad_signal(mic, padding, padding)
        farend = pad_signal(farend, padding, padding)
        states_one = self._initial_state(self.input_one, 2)
        states_two = self._initial_state(self.input_two, 2)
        output = np.zeros(len(mic), dtype=np.float32)
        mic_buffer = np.zeros(self.audio.block_len, dtype=np.float32)
        farend_buffer = np.zeros(self.audio.block_len, dtype=np.float32)
        out_buffer = np.zeros(self.audio.block_len, dtype=np.float32)

        total_blocks = (len(mic) - padding) // self.audio.block_shift
        for idx in range(total_blocks):
            mic_buffer[:-self.audio.block_shift] = mic_buffer[self.audio.block_shift :]
            farend_buffer[:-self.audio.block_shift] = farend_buffer[self.audio.block_shift :]
            mic_buffer[-self.audio.block_shift :] = mic[
                idx * self.audio.block_shift : (idx * self.audio.block_shift) + self.audio.block_shift
            ]
            farend_buffer[-self.audio.block_shift :] = farend[
                idx * self.audio.block_shift : (idx * self.audio.block_shift) + self.audio.block_shift
            ]

            mic_mag, mic_phase = self.signal.numpy_rfft(mic_buffer)
            farend_mag, _ = self.signal.numpy_rfft(farend_buffer)
            mic_mag = np.reshape(mic_mag, (1, 1, -1)).astype(np.float32)
            farend_mag = np.reshape(farend_mag, (1, 1, -1)).astype(np.float32)

            self.stage_one.set_tensor(self.input_one[0]["index"], mic_mag)
            self.stage_one.set_tensor(self.input_one[1]["index"], farend_mag)
            self.stage_one.set_tensor(self.input_one[2]["index"], states_one)
            self.stage_one.invoke()
            masked_mag = self.stage_one.get_tensor(self.output_one[0]["index"])
            states_one = self.stage_one.get_tensor(self.output_one[1]["index"])

            estimated_frame = self.signal.numpy_irfft(np.squeeze(masked_mag), mic_phase)
            estimated_frame = np.reshape(estimated_frame, (1, 1, -1)).astype(np.float32)
            farend_frame = np.reshape(farend_buffer, (1, 1, -1)).astype(np.float32)

            self.stage_two.set_tensor(self.input_two[0]["index"], estimated_frame)
            self.stage_two.set_tensor(self.input_two[1]["index"], farend_frame)
            self.stage_two.set_tensor(self.input_two[2]["index"], states_two)
            self.stage_two.invoke()
            out_block = self.stage_two.get_tensor(self.output_two[0]["index"])
            states_two = self.stage_two.get_tensor(self.output_two[1]["index"])

            out_buffer[:-self.audio.block_shift] = out_buffer[self.audio.block_shift :]
            out_buffer[-self.audio.block_shift :] = 0.0
            out_buffer += np.squeeze(out_block)
            output[idx * self.audio.block_shift : (idx * self.audio.block_shift) + self.audio.block_shift] = out_buffer[
                : self.audio.block_shift
            ]

        return output[padding : padding + len(mic) - padding * 2]


def enhance_folder(
    audio: AudioConfig,
    model: ModelConfig,
    inference: InferenceConfig,
    input_root: str | Path,
    output_root: str | Path,
) -> list[Path]:
    source = Path(input_root).expanduser().resolve()
    target = ensure_directory(output_root)
    outputs: list[Path] = []

    if inference.tflite:
        runner = TFLiteAECEnhancer(audio, inference.model_prefix)
        for path in sorted(source.rglob(f"*{inference.mic_suffix}")):
            farend = Path(str(path).replace(inference.mic_suffix, inference.lpb_suffix))
            mic_audio = read_mono_audio(path, expected_sample_rate=audio.sample_rate).data
            farend_audio = read_mono_audio(farend, expected_sample_rate=audio.sample_rate).data
            enhanced = runner.process(mic_audio, farend_audio)
            destination = ensure_directory((target / path.relative_to(source)).parent) / path.name
            sf.write(str(destination), clip_audio(enhanced), audio.sample_rate)
            outputs.append(destination)
        return outputs

    runner = OfflineEnhancer(audio, model, inference)
    pattern = "*.wav" if model.mode == "denoise" else f"*{inference.mic_suffix}"
    for path in sorted(source.rglob(pattern)):
        destination = ensure_directory((target / path.relative_to(source)).parent) / path.name
        outputs.append(runner.process_file(path, destination))
    return outputs
