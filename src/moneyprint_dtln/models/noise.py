"""Single-input DTLN denoising model builder."""

from __future__ import annotations

from typing import Any

from ..config import AudioConfig, ModelConfig
from ..layers import InstantLayerNormalization
from ..tf_compat import require_tensorflow
from .common import create_context, loss_wrapper, maybe_log_norm, multiply_layer, separation_kernel


class DTLNNoiseModel:
    def __init__(self, audio: AudioConfig, model: ModelConfig) -> None:
        self.audio = audio
        self.config = model
        self.context = create_context(audio, model)

    def build_training_model(self) -> Any:
        tf = require_tensorflow()
        layers = tf.keras.layers
        signal = self.context.signal

        time_dat = layers.Input(batch_shape=(None, None))
        stft = signal.stft(time_dat)
        magnitude = stft.magnitude
        angle = stft.phase

        norm_mag = maybe_log_norm(magnitude, self.config.use_log_stft_norm)
        mask_1 = separation_kernel(norm_mag, (self.audio.block_len // 2 + 1), self.config)
        estimated_mag = multiply_layer(self.config)([magnitude, mask_1])
        estimated_frames = layers.Lambda(lambda x: signal.ifft(x[0], x[1]))([estimated_mag, angle])

        encoded = layers.Conv1D(self.config.encoder_size, 1, strides=1, use_bias=False)(estimated_frames)
        encoded_norm = InstantLayerNormalization()(encoded)
        mask_2 = separation_kernel(encoded_norm, self.config.encoder_size, self.config)
        estimated = multiply_layer(self.config)([encoded, mask_2])
        decoded = layers.Conv1D(self.audio.block_len, 1, padding="causal", use_bias=False)(estimated)
        enhanced = layers.Lambda(signal.overlap_add)(decoded)
        return tf.keras.Model(inputs=time_dat, outputs=enhanced, name="dtln_noise_train")

    def build_stateful_model(self) -> Any:
        tf = require_tensorflow()
        layers = tf.keras.layers
        signal = self.context.signal

        time_dat = layers.Input(batch_shape=(1, self.audio.block_len))
        fft = signal.fft_frame(time_dat)
        magnitude = fft.magnitude
        angle = fft.phase

        norm_mag = maybe_log_norm(magnitude, self.config.use_log_stft_norm)
        mask_1 = separation_kernel(
            norm_mag,
            (self.audio.block_len // 2 + 1),
            self.config,
            stateful=True,
        )
        estimated_mag = multiply_layer(self.config)([magnitude, mask_1])
        estimated_frames = layers.Lambda(lambda x: signal.ifft(x[0], x[1]))([estimated_mag, angle])
        encoded = layers.Conv1D(self.config.encoder_size, 1, strides=1, use_bias=False)(estimated_frames)
        encoded_norm = InstantLayerNormalization()(encoded)
        mask_2 = separation_kernel(
            encoded_norm,
            self.config.encoder_size,
            self.config,
            stateful=True,
        )
        estimated = multiply_layer(self.config)([encoded, mask_2])
        decoded = layers.Conv1D(self.audio.block_len, 1, padding="causal", use_bias=False)(estimated)
        return tf.keras.Model(inputs=time_dat, outputs=decoded, name="dtln_noise_stateful")

    def compile(self, model: Any) -> Any:
        tf = require_tensorflow()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=3.0)
        model.compile(loss=loss_wrapper(), optimizer=optimizer)
        return model
