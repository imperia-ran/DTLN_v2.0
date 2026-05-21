"""Legacy TFLite evaluation wrapper for the new runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moneyprint_dtln.config import AudioConfig, InferenceConfig, ModelConfig
from moneyprint_dtln.runtime import enhance_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="legacy AEC TFLite runner")
    parser.add_argument("--in_folder", "-i", required=True, help="folder with mic/lpb wav files")
    parser.add_argument("--out_folder", "-o", required=True, help="folder for enhanced wav files")
    parser.add_argument("--model", "-m", required=True, help="tflite model prefix without _1/_2 suffix")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio = AudioConfig(sample_rate=16000, block_len=512, block_shift=256)
    model = ModelConfig(
        mode="aec",
        lstm_units=64,
        lstm_layers=2,
        encoder_size=192,
        use_log_stft_norm=True,
        use_safe_multiply=True,
    )
    inference = InferenceConfig(
        model_prefix=Path(args.model),
        input_root=Path(args.in_folder),
        output_root=Path(args.out_folder),
        tflite=True,
        mic_suffix="mic.wav",
        lpb_suffix="lpb.wav",
    )
    enhance_folder(audio, model, inference, args.in_folder, args.out_folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
