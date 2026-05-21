"""Custom Keras layers used by the DTLN model family."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - import depends on environment
    import tensorflow as _tf  # type: ignore
except Exception:  # pragma: no cover - import depends on environment
    _tf = None


BaseLayer = _tf.keras.layers.Layer if _tf is not None else object


class InstantLayerNormalization(BaseLayer):
    """Frame-wise normalization as used in DTLN-style models."""

    def __init__(self, epsilon: float = 1e-7, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None

    def build(self, input_shape: Any) -> None:
        shape = input_shape[-1:]
        self.gamma = self.add_weight(
            shape=shape,
            initializer="ones",
            trainable=True,
            name="gamma",
        )
        self.beta = self.add_weight(
            shape=shape,
            initializer="zeros",
            trainable=True,
            name="beta",
        )

    def call(self, inputs: Any) -> Any:
        tf = require_tensorflow()
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean), axis=[-1], keepdims=True)
        std = tf.math.sqrt(variance + self.epsilon)
        normalized = (inputs - mean) / std
        return normalized * self.gamma + self.beta


class SafeMultiply(BaseLayer):
    """Optional clipping before multiplication to reduce numerical spikes."""

    def __init__(self, clip_value: float = 10000.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.clip_value = clip_value

    def call(self, inputs: Any) -> Any:
        tf = require_tensorflow()
        left, right = inputs
        left = tf.clip_by_value(left, -self.clip_value, self.clip_value)
        right = tf.clip_by_value(right, -self.clip_value, self.clip_value)
        return left * right
