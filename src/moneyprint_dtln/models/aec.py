"""Dual-input DTLN-based acoustic echo cancellation model builder."""

from __future__ import annotations

from typing import Any

from ..config import AudioConfig, ModelConfig
from ..layers import InstantLayerNormalization
from ..tf_compat import require_tensorflow
from .common import create_context, loss_wrapper, maybe_log_norm, multiply_layer, separation_kernel


class DTLNAECModel:
    def __init__(self, audio: AudioConfig, model: ModelConfig) -> None:
        self.audio = audio
        self.config = model
        self.context = create_context(audio, model)

    def _split_inputs(self, time_dat: Any) -> tuple[Any, Any]:
        tf = require_tensorflow()
        half = tf.shape(time_dat)[-1] // 2
        mic = time_dat[:, :half]
        farend = time_dat[:, half:]
        return mic, farend

    def build_training_model(self) -> Any:
        tf = require_tensorflow()
        layers = tf.keras.layers
        signal = self.context.signal

        time_dat = layers.Input(batch_shape=(None, None))
        mic_data, farend_data = self._split_inputs(time_dat)

        farend_frames = layers.Lambda(signal.frame)(farend_data)
        mic_fft = signal.stft(mic_data)
        farend_fft = signal.stft(farend_data)

        mic_norm = maybe_log_norm(mic_fft.magnitude, self.config.use_log_stft_norm)
        farend_norm = maybe_log_norm(farend_fft.magnitude, self.config.use_log_stft_norm)
        combined_mag = layers.Concatenate(axis=-1)([mic_norm, farend_norm])

        mask_1 = separation_kernel(combined_mag, (self.audio.block_len // 2 + 1), self.config)
        estimated_mag = multiply_layer(self.config)([mic_fft.magnitude, mask_1])
        estimated_frames = layers.Lambda(lambda x: signal.ifft(x[0], x[1]))([estimated_mag, mic_fft.phase])

        encoded_mic = layers.Conv1D(self.config.encoder_size, 1, strides=1, use_bias=False)(estimated_frames)
        encoded_farend = layers.Conv1D(self.config.encoder_size, 1, strides=1, use_bias=False)(farend_frames)
        encoded_mic_norm = InstantLayerNormalization()(encoded_mic)
        encoded_farend_norm = InstantLayerNormalization()(encoded_farend)
        encoded = layers.Concatenate(axis=-1)([encoded_mic_norm, encoded_farend_norm])

        mask_2 = separation_kernel(encoded, self.config.encoder_size, self.config)
        estimated = multiply_layer(self.config)([encoded_mic, mask_2])
        decoded = layers.Conv1D(self.audio.block_len, 1, padding="causal", use_bias=False)(estimated)
        enhanced = layers.Lambda(signal.overlap_add)(decoded)
        return tf.keras.Model(inputs=time_dat, outputs=enhanced, name="dtln_aec_train")

    def build_stateful_model(self) -> Any:
        tf = require_tensorflow()
        layers = tf.keras.layers
        signal = self.context.signal

        mic_input = layers.Input(batch_shape=(1, self.audio.block_len), name="mic")
        farend_input = layers.Input(batch_shape=(1, self.audio.block_len), name="farend")

        farend_frames = layers.Lambda(signal.frame)(farend_input)
        mic_fft = signal.stft(mic_input)
        farend_fft = signal.stft(farend_input)

        mic_norm = maybe_log_norm(mic_fft.magnitude, self.config.use_log_stft_norm)
        farend_norm = maybe_log_norm(farend_fft.magnitude, self.config.use_log_stft_norm)
        combined_mag = layers.Concatenate(axis=-1)([mic_norm, farend_norm])

        mask_1 = separation_kernel(
            combined_mag,
            (self.audio.block_len // 2 + 1),
            self.config,
            stateful=True,
        )
        estimated_mag = multiply_layer(self.config)([mic_fft.magnitude, mask_1])
        estimated_frames = layers.Lambda(lambda x: signal.ifft(x[0], x[1]))([estimated_mag, mic_fft.phase])

        encoded_mic = layers.Conv1D(self.config.encoder_size, 1, strides=1, use_bias=False)(estimated_frames)
        encoded_farend = layers.Conv1D(self.config.encoder_size, 1, strides=1, use_bias=False)(farend_frames)
        encoded_mic_norm = InstantLayerNormalization()(encoded_mic)
        encoded_farend_norm = InstantLayerNormalization()(encoded_farend)
        encoded = layers.Concatenate(axis=-1)([encoded_mic_norm, encoded_farend_norm])

        mask_2 = separation_kernel(
            encoded,
            self.config.encoder_size,
            self.config,
            stateful=True,
        )
        estimated = multiply_layer(self.config)([encoded_mic, mask_2])
        decoded = layers.Conv1D(self.audio.block_len, 1, padding="causal", use_bias=False)(estimated)
        return tf.keras.Model(inputs=[mic_input, farend_input], outputs=decoded, name="dtln_aec_stateful")

    def build_tflite_pair(self) -> tuple[Any, Any]:
        tf = require_tensorflow()
        layers = tf.keras.layers
        signal = self.context.signal

        mic_mag = layers.Input(batch_shape=(1, 1, (self.audio.block_len // 2 + 1)), name="mic_mag")
        farend_mag = layers.Input(batch_shape=(1, 1, (self.audio.block_len // 2 + 1)), name="farend_mag")
        states_in_1 = layers.Input(
            batch_shape=(1, self.config.lstm_layers, self.config.lstm_units, 2),
            name="states_1",
        )
        mic_norm = maybe_log_norm(mic_mag, self.config.use_log_stft_norm)
        farend_norm = maybe_log_norm(farend_mag, self.config.use_log_stft_norm)
        combined_mag = layers.Concatenate(axis=-1)([mic_norm, farend_norm])
        mask_1, states_out_1 = separation_kernel(
            combined_mag,
            (self.audio.block_len // 2 + 1),
            self.config,
            return_states=True,
            in_states=states_in_1,
        )
        masked_mag = multiply_layer(self.config)([mic_mag, mask_1])
        model_1 = tf.keras.Model(
            inputs=[mic_mag, farend_mag, states_in_1],
            outputs=[masked_mag, states_out_1],
            name="dtln_aec_tflite_stage_1",
        )

        estimated_frame = layers.Input(batch_shape=(1, 1, self.audio.block_len), name="estimated_frame")
        farend_frame = layers.Input(batch_shape=(1, 1, self.audio.block_len), name="farend_frame")
        states_in_2 = layers.Input(
            batch_shape=(1, self.config.lstm_layers, self.config.lstm_units, 2),
            name="states_2",
        )
        encoded_mic = layers.Conv1D(self.config.encoder_size, 1, strides=1, use_bias=False)(estimated_frame)
        encoded_farend = layers.Conv1D(self.config.encoder_size, 1, strides=1, use_bias=False)(farend_frame)
        encoded_mic_norm = InstantLayerNormalization()(encoded_mic)
        encoded_farend_norm = InstantLayerNormalization()(encoded_farend)
        stage_two_input = layers.Concatenate(axis=-1)([encoded_mic_norm, encoded_farend_norm])
        mask_2, states_out_2 = separation_kernel(
            stage_two_input,
            self.config.encoder_size,
            self.config,
            return_states=True,
            in_states=states_in_2,
        )
        estimated = multiply_layer(self.config)([encoded_mic, mask_2])
        decoded = layers.Conv1D(self.audio.block_len, 1, padding="causal", use_bias=False)(estimated)
        model_2 = tf.keras.Model(
            inputs=[estimated_frame, farend_frame, states_in_2],
            outputs=[decoded, states_out_2],
            name="dtln_aec_tflite_stage_2",
        )
        return model_1, model_2

    def compile(self, model: Any, learning_rate: float, clip_norm: float) -> Any:
        tf = require_tensorflow()
        if self.config.mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clip_norm)
        model.compile(loss=loss_wrapper(), optimizer=optimizer)
        return model
