from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def write_wav(path: Path, data: np.ndarray, sample_rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float32), sample_rate)
    return path


def sine(length: int, scale: float = 0.3, frequency: float = 220.0, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / sample_rate
    return scale * np.sin(2.0 * np.pi * frequency * t).astype(np.float32)
