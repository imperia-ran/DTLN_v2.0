"""Lazy TensorFlow import helpers."""

from __future__ import annotations

from typing import Any

from .exceptions import TensorFlowUnavailableError


def require_tensorflow() -> Any:
    """Import TensorFlow on demand and raise a clearer package error on failure."""

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise TensorFlowUnavailableError(
            "TensorFlow is required for this operation. Install the "
            "`train` or `export` extras, or provision a TensorFlow runtime."
        ) from exc
    return tf
