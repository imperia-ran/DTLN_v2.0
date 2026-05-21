"""Legacy entry point kept for compatibility with the original repository layout."""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moneyprint_dtln.legacy import DTLN_aecmodel, DTLN_model


def _first_weight_file(directory: str | Path) -> Path:
    path = Path(directory).expanduser().resolve()
    matches = fnmatch.filter([item.name for item in path.iterdir() if item.is_file()], "*.weights.h5")
    if not matches:
        raise FileNotFoundError(f"no .weights.h5 file found in {path}")
    return path / matches[0]


def _train_aec() -> None:
    dataset_root = Path("/path/to/dataset")
    mic_train = dataset_root / "train" / "nearend_mic_signal"
    farend_train = dataset_root / "train" / "farend_speech"
    clean_train = dataset_root / "train" / "nearend_speech"
    mic_val = dataset_root / "val" / "nearend_mic_signal"
    farend_val = dataset_root / "val" / "farend_speech"
    clean_val = dataset_root / "val" / "nearend_speech"

    model = DTLN_aecmodel()
    model.build_DTLN_model()
    model.compile_model()
    model.train_model(
        "dtln_aec_legacy_wrapper",
        str(mic_train),
        str(farend_train),
        str(clean_train),
        str(mic_val),
        str(farend_val),
        str(clean_val),
    )


def _continue_train(weights_prefix: str) -> None:
    dataset_root = Path("/path/to/dataset")
    mic_train = dataset_root / "train" / "nearend_mic_signal"
    farend_train = dataset_root / "train" / "farend_speech"
    clean_train = dataset_root / "train" / "nearend_speech"
    mic_val = dataset_root / "val" / "nearend_mic_signal"
    farend_val = dataset_root / "val" / "farend_speech"
    clean_val = dataset_root / "val" / "nearend_speech"

    model = DTLN_aecmodel()
    model.build_DTLN_model()
    model.model.load_weights(f"{weights_prefix}.weights.h5")
    model.compile_model()
    model.train_model(
        "dtln_aec_legacy_wrapper",
        str(mic_train),
        str(farend_train),
        str(clean_train),
        str(mic_val),
        str(farend_val),
        str(clean_val),
    )


def _convert(directory: str) -> None:
    weight_file = _first_weight_file(directory)
    prefix = weight_file.with_suffix("").with_suffix("")
    model = DTLN_aecmodel()
    model.create_tf_lite_model(str(weight_file), str(prefix), use_dynamic_range_quant=True)


def _save_stateful(weights_prefix: str) -> None:
    model = DTLN_aecmodel()
    model.build_DTLN_model_stateful()
    model.model.load_weights(f"{weights_prefix}.weights.h5")
    model.create_saved_model(f"{weights_prefix}.weights.h5", f"{weights_prefix}_saved_model")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python DTLN_model.py [train|contrain|convert|param] ...")
        return 1

    command = argv[0]
    if command == "train":
        _train_aec()
        return 0
    if command == "contrain":
        _continue_train(argv[1])
        return 0
    if command == "convert":
        _convert(argv[1])
        return 0
    if command == "param":
        _save_stateful(argv[1])
        return 0

    print(f"unknown command: {command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
