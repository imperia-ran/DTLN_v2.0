"""Command line interface for the rewritten DTLN/AEC project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ProjectConfig
from .data.manifest import write_manifest
from .data.pairs import build_manifest_bundle
from .exporters import ExportService
from .inspection import format_dataset_summary, format_model_summary, inspect_dataset, inspect_model_builder, save_summary
from .metrics import evaluate
from .runtime import enhance_folder
from .training import Trainer
from .utils.audio import read_mono_audio
from .utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="moneyprint DTLN toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="build a dataset manifest")
    manifest.add_argument("--config", help="config JSON file", required=True)
    manifest.add_argument("--root", help="logical dataset root", required=True)
    manifest.add_argument("--output", help="manifest JSON path", required=True)

    train = subparsers.add_parser("train", help="train a model")
    train.add_argument("--config", help="config JSON file", required=True)

    export = subparsers.add_parser("export", help="export weights to deployment formats")
    export.add_argument("--config", help="config JSON file", required=True)

    enhance = subparsers.add_parser("enhance", help="run offline inference on a folder")
    enhance.add_argument("--config", help="config JSON file", required=True)
    enhance.add_argument("--input", help="input folder override")
    enhance.add_argument("--output", help="output folder override")

    inspect = subparsers.add_parser("inspect", help="inspect dataset and model settings")
    inspect.add_argument("--config", help="config JSON file", required=True)
    inspect.add_argument("--output", help="optional report file path")

    score = subparsers.add_parser("score", help="score one estimate against a reference")
    score.add_argument("--reference", required=True)
    score.add_argument("--estimate", required=True)
    score.add_argument("--frame-size", type=int, default=512)

    dump = subparsers.add_parser("dump-config", help="write a default config template")
    dump.add_argument("--output", help="config JSON path", required=True)
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    if args.command == "dump-config":
        config = ProjectConfig()
        Path(args.output).write_text(json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8")
        return 0

    config = ProjectConfig.from_json(args.config)

    if args.command == "manifest":
        bundle = build_manifest_bundle(args.root, config.dataset)
        write_manifest(args.output, bundle)
        return 0

    if args.command == "train":
        Trainer(config).train()
        return 0

    if args.command == "export":
        ExportService(config).export()
        return 0

    if args.command == "enhance":
        input_root = args.input or config.inference.input_root
        output_root = args.output or config.inference.output_root
        if input_root is None or output_root is None:
            raise ValueError("input and output roots must be provided for enhance")
        enhance_folder(config.audio, config.model, config.inference, input_root, output_root)
        return 0

    if args.command == "inspect":
        dataset_summary = inspect_dataset(config)
        model_summary = inspect_model_builder(config.audio, config.model)
        text = "\n\n".join(
            [
                "[dataset]",
                format_dataset_summary(dataset_summary),
                "[model]",
                format_model_summary(model_summary),
            ]
        )
        if args.output:
            save_summary(text, args.output)
        else:
            print(text)
        return 0

    if args.command == "score":
        reference = read_mono_audio(args.reference).data
        estimate = read_mono_audio(args.estimate).data
        result = evaluate(reference, estimate, frame_size=args.frame_size)
        print(json.dumps(result.__dict__, indent=2))
        return 0

    raise ValueError(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
