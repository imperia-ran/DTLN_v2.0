"""Shared model-building helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import AudioConfig, ModelConfig
from ..layers import InstantLayerNormalization, SafeMultiply
from ..signal import SignalTransform
from ..tf_compat import require_tensorflow


@dataclass(slots=True)
class ModelContext:
    audio: AudioConfig
    model: ModelConfig
    signal: SignalTransform


def create_context(audio: AudioConfig, model: ModelConfig) -> ModelContext:
    return ModelContext(audio=audio, model=model, signal=SignalTransform(audio.block_len, audio.block_shift))


def snr_cost(s_estimate: Any, s_true: Any) -> Any:
    tf = require_tensorflow()
    snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / (
        tf.reduce_mean(tf.math.square(s_true - s_estimate), axis=-1, keepdims=True) + 1e-7
    )
    numerator = tf.math.log(snr)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return -10 * (numerator / denominator)


def loss_wrapper() -> Any:
    def loss_fn(y_true: Any, y_pred: Any) -> Any:
        tf = require_tensorflow()
        loss = tf.squeeze(snr_cost(y_pred, y_true))
        return tf.reduce_mean(loss)

    return loss_fn


def maybe_log_norm(tensor: Any, enabled: bool) -> Any:
    tf = require_tensorflow()
    if not enabled:
        return tensor
    return InstantLayerNormalization()(tf.math.log(tensor + 1e-7))


def multiply_layer(model_config: ModelConfig):
    tf = require_tensorflow()
    if model_config.use_safe_multiply:
        return SafeMultiply(clip_value=model_config.clip_multiply)
    return tf.keras.layers.Multiply()


def separation_kernel(
    x: Any,
    mask_size: int,
    config: ModelConfig,
    stateful: bool = False,
    return_states: bool = False,
    in_states: Any | None = None,
) -> Any:
    tf = require_tensorflow()
    layers = tf.keras.layers
    states_h = []
    states_c = []
    for index in range(config.lstm_layers):
        if return_states:
            if in_states is None:
                raise ValueError("in_states is required when return_states=True")
            initial_state = [in_states[:, index, :, 0], in_states[:, index, :, 1]]
            x, h_state, c_state = layers.LSTM(
                config.lstm_units,
                return_sequences=True,
                return_state=True,
                stateful=stateful,
                unroll=True,
            )(x, initial_state=initial_state)
            states_h.append(h_state)
            states_c.append(c_state)
        else:
            x = layers.LSTM(
                config.lstm_units,
                return_sequences=True,
                stateful=stateful,
            )(x)
        if index < config.lstm_layers - 1:
            x = layers.Dropout(config.dropout)(x)
    mask = layers.Dense(mask_size)(x)
    mask = layers.Activation(config.activation)(mask)
    if not return_states:
        return mask
    out_states_h = tf.reshape(tf.stack(states_h, axis=0), [1, config.lstm_layers, config.lstm_units])
    out_states_c = tf.reshape(tf.stack(states_c, axis=0), [1, config.lstm_layers, config.lstm_units])
    out_states = tf.stack([out_states_h, out_states_c], axis=-1)
    return mask, out_states
