"""Command-line interface for validation, replay, splits, vocabulary, and metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset_manifest import (
    build_prefill_dataset_manifest,
    validate_prefill_dataset_manifest,
    verify_prefill_dataset_manifest,
)
from .metrics import compute_dasr, compute_pasr
from .legacy_npz import adapt_prefill_rank_npz
from .replay import replay_records
from .schema import (
    ValidationError,
    adapt_legacy_manifest,
    load_jsonl,
    validate_trace_records,
    write_jsonl,
)
from .splits import assign_grouped_splits, freeze_vocabulary, validate_frozen_vocabulary


def _write_json(path: str, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-traces")
    validate.add_argument("--input", required=True)
    validate.add_argument("--group-field", default="case_id")

    adapt = commands.add_parser("adapt-legacy-manifest")
    adapt.add_argument("--input", required=True)
    adapt.add_argument("--output", required=True)

    adapt_npz = commands.add_parser("adapt-prefill-rank-npz")
    adapt_npz.add_argument("--input", required=True)
    adapt_npz.add_argument("--legacy-path", required=True)
    adapt_npz.add_argument("--case-id", required=True)
    adapt_npz.add_argument("--output", required=True)

    build_manifest = commands.add_parser("build-prefill-dataset-manifest")
    build_manifest.add_argument("--dataset-id", required=True)
    build_manifest.add_argument("--csv", required=True)
    build_manifest.add_argument("--csv-legacy-path", required=True)
    build_manifest.add_argument("--script", required=True)
    build_manifest.add_argument("--script-legacy-path", required=True)
    build_manifest.add_argument("--pointer", required=True)
    build_manifest.add_argument("--payload", required=True)
    build_manifest.add_argument("--payload-legacy-path", required=True)
    build_manifest.add_argument("--output", required=True)

    validate_manifest = commands.add_parser("validate-prefill-dataset-manifest")
    validate_manifest.add_argument("--input", required=True)
    validate_manifest.add_argument("--csv", required=True)
    validate_manifest.add_argument("--script", required=True)
    validate_manifest.add_argument("--pointer", required=True)
    validate_manifest.add_argument("--payload", required=True)

    replay = commands.add_parser("replay")
    replay.add_argument("--input", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--external-payload")

    split = commands.add_parser("split")
    split.add_argument("--input", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--group-field", default="case_id")
    split.add_argument("--seed", default="janus-v1")
    split.add_argument("--train-ratio", type=float, default=0.7)
    split.add_argument("--validation-ratio", type=float, default=0.15)
    split.add_argument("--test-ratio", type=float, default=0.15)

    vocabulary = commands.add_parser("freeze-vocabulary")
    vocabulary.add_argument("--input", required=True)
    vocabulary.add_argument("--output", required=True)

    metrics = commands.add_parser("metrics")
    metrics.add_argument("--pasr")
    metrics.add_argument("--dasr")
    metrics.add_argument("--vocabulary")
    metrics.add_argument("--output")

    qai_inspect = commands.add_parser("qai-inspect-feature")
    qai_inspect.add_argument("--input", required=True)
    qai_inspect.add_argument("--selected-ranks", type=int, default=10)

    qai_train = commands.add_parser("qai-train")
    qai_train.add_argument("--manifest", required=True)
    qai_train.add_argument("--config", required=True)
    qai_train.add_argument("--checkpoint-dir", required=True)
    qai_train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    qai_infer = commands.add_parser("qai-infer")
    qai_infer.add_argument("--manifest", required=True)
    qai_infer.add_argument("--checkpoint-dir", required=True)
    qai_infer.add_argument("--split", choices=("train", "validation", "test"), default="test")
    qai_infer.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    qai_infer.add_argument("--output", required=True)
    qai_infer.add_argument("--metrics-output", required=True)

    qai_smoke = commands.add_parser("qai-synthetic-smoke")
    qai_smoke.add_argument("--work-dir", required=True)
    qai_smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    atr_validate = commands.add_parser("atr-validate")
    atr_validate.add_argument("--manifest", required=True)

    atr_split = commands.add_parser("atr-split")
    atr_split.add_argument("--manifest", required=True)
    atr_split.add_argument("--output", required=True)
    atr_split.add_argument("--seed", default="janus-atr-v1")
    atr_split.add_argument("--train-ratio", type=float, default=0.8)
    atr_split.add_argument("--validation-ratio", type=float, default=0.1)
    atr_split.add_argument("--test-ratio", type=float, default=0.1)

    atr_vocabulary = commands.add_parser("atr-freeze-vocabulary")
    atr_vocabulary.add_argument("--manifest", required=True)
    atr_vocabulary.add_argument("--output", required=True)

    atr_train = commands.add_parser("atr-train")
    atr_train.add_argument("--manifest", required=True)
    atr_train.add_argument("--config", required=True)
    atr_train.add_argument("--checkpoint-dir", required=True)
    atr_train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    atr_infer = commands.add_parser("atr-infer")
    atr_infer.add_argument("--manifest", required=True)
    atr_infer.add_argument("--checkpoint-dir", required=True)
    atr_infer.add_argument("--split", choices=("train", "validation", "test"), default="test")
    atr_infer.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    atr_infer.add_argument("--output", required=True)
    atr_infer.add_argument("--metrics-output", required=True)
    atr_infer.add_argument("--no-evaluate", action="store_true")

    atr_smoke = commands.add_parser("atr-synthetic-smoke")
    atr_smoke.add_argument("--work-dir", required=True)
    atr_smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def run(arguments: argparse.Namespace) -> dict:
    if arguments.command == "validate-traces":
        records = validate_trace_records(
            load_jsonl(arguments.input), group_field=arguments.group_field
        )
        return {"status": "ok", "trace_records": len(records)}
    if arguments.command == "adapt-legacy-manifest":
        records = adapt_legacy_manifest(load_jsonl(arguments.input))
        write_jsonl(arguments.output, records)
        return {"status": "ok", "adapted_records": len(records), "output": arguments.output}
    if arguments.command == "adapt-prefill-rank-npz":
        record = adapt_prefill_rank_npz(
            arguments.input,
            legacy_path=arguments.legacy_path,
            case_id=arguments.case_id,
        )
        write_jsonl(arguments.output, [record])
        return {
            "status": "ok",
            "adapted_records": 1,
            "evaluation_eligible": False,
            "output": arguments.output,
        }
    if arguments.command == "build-prefill-dataset-manifest":
        manifest = build_prefill_dataset_manifest(
            dataset_id=arguments.dataset_id,
            csv_path=arguments.csv,
            csv_legacy_path=arguments.csv_legacy_path,
            script_path=arguments.script,
            script_legacy_path=arguments.script_legacy_path,
            pointer_path=arguments.pointer,
            payload_path=arguments.payload,
            payload_legacy_path=arguments.payload_legacy_path,
        )
        _write_json(arguments.output, manifest)
        return {
            "status": "ok",
            "manifest_entries": len(manifest["entries"]),
            "split": "unassigned",
            "output": arguments.output,
        }
    if arguments.command == "validate-prefill-dataset-manifest":
        manifest = json.loads(Path(arguments.input).read_text(encoding="utf-8"))
        validate_prefill_dataset_manifest(manifest)
        verify_prefill_dataset_manifest(
            manifest,
            csv_path=arguments.csv,
            script_path=arguments.script,
            pointer_path=arguments.pointer,
            payload_path=arguments.payload,
        )
        return {
            "status": "ok",
            "manifest_entries": len(manifest["entries"]),
            "source_bytes_verified": True,
        }
    if arguments.command == "replay":
        records = replay_records(
            load_jsonl(arguments.input),
            external_payload_path=arguments.external_payload,
        )
        write_jsonl(arguments.output, records)
        return {"status": "ok", "replayed_records": len(records), "output": arguments.output}
    if arguments.command == "split":
        records = assign_grouped_splits(
            load_jsonl(arguments.input),
            group_field=arguments.group_field,
            train_ratio=arguments.train_ratio,
            validation_ratio=arguments.validation_ratio,
            test_ratio=arguments.test_ratio,
            seed=arguments.seed,
        )
        validate_trace_records(records, group_field=arguments.group_field)
        write_jsonl(arguments.output, records)
        return {"status": "ok", "split_records": len(records), "output": arguments.output}
    if arguments.command == "freeze-vocabulary":
        vocabulary = freeze_vocabulary(load_jsonl(arguments.input))
        _write_json(arguments.output, vocabulary)
        return {"status": "ok", "vocabulary_size": len(vocabulary["tokens"]), "output": arguments.output}
    if arguments.command == "metrics":
        if not arguments.pasr and not arguments.dasr:
            raise ValidationError("metrics requires --pasr and/or --dasr")
        result: dict = {}
        if arguments.pasr:
            result["pasr"] = compute_pasr(load_jsonl(arguments.pasr))
        if arguments.dasr:
            frozen = None
            if arguments.vocabulary:
                vocabulary = json.loads(Path(arguments.vocabulary).read_text(encoding="utf-8"))
                frozen = validate_frozen_vocabulary(vocabulary)
            result["dasr"] = compute_dasr(
                load_jsonl(arguments.dasr), frozen_vocabulary=frozen
            )
        if arguments.output:
            _write_json(arguments.output, result)
        return result
    if arguments.command == "qai-inspect-feature":
        from .qai import prefill_rank_feature

        _, metadata = prefill_rank_feature(
            arguments.input, selected_ranks=arguments.selected_ranks
        )
        return {"status": "ok", **metadata}
    if arguments.command == "qai-train":
        from .qai import QAIConfig, load_qai_records, train_qai

        return train_qai(
            load_qai_records(arguments.manifest),
            QAIConfig.from_json(arguments.config),
            arguments.checkpoint_dir,
            device=arguments.device,
        )
    if arguments.command == "qai-infer":
        from .qai import load_qai_records, predict_qai, write_qai_predictions

        predictions, metrics = predict_qai(
            load_qai_records(arguments.manifest),
            arguments.checkpoint_dir,
            split=arguments.split,
            device=arguments.device,
        )
        write_qai_predictions(arguments.output, predictions)
        _write_json(arguments.metrics_output, metrics)
        return {
            "status": "ok",
            "predictions": len(predictions),
            "output": arguments.output,
            "metrics_output": arguments.metrics_output,
        }
    if arguments.command == "qai-synthetic-smoke":
        from .qai import run_synthetic_qai_smoke

        return run_synthetic_qai_smoke(arguments.work_dir, device=arguments.device)
    if arguments.command == "atr-validate":
        from .atr_data import load_atr_records

        records = load_atr_records(arguments.manifest)
        return {
            "status": "ok", "responses": len(records),
            "steps": sum(len(record["steps"]) for record in records),
        }
    if arguments.command == "atr-split":
        from .atr_data import assign_atr_grouped_splits, load_atr_records

        records = assign_atr_grouped_splits(
            load_atr_records(arguments.manifest),
            train_ratio=arguments.train_ratio,
            validation_ratio=arguments.validation_ratio,
            test_ratio=arguments.test_ratio,
            seed=arguments.seed,
        )
        write_jsonl(arguments.output, records)
        return {"status": "ok", "split_responses": len(records), "output": arguments.output}
    if arguments.command == "atr-freeze-vocabulary":
        from .atr_data import freeze_atr_vocabulary, load_atr_records

        vocabulary = freeze_atr_vocabulary(load_atr_records(arguments.manifest))
        _write_json(arguments.output, vocabulary)
        return {
            "status": "ok", "vocabulary_size": len(vocabulary["token_ids"]),
            "source_split": "train", "output": arguments.output,
        }
    if arguments.command == "atr-train":
        from .atr import train_atr
        from .atr_data import load_atr_records
        from .atr_features import ATRConfig

        return train_atr(
            load_atr_records(arguments.manifest),
            ATRConfig.from_json(arguments.config),
            arguments.checkpoint_dir,
            device=arguments.device,
            run_kind="unvalidated_research_run",
        )
    if arguments.command == "atr-infer":
        from .atr import predict_atr
        from .atr_data import load_atr_records

        evaluate = not arguments.no_evaluate
        predictions, metrics = predict_atr(
            load_atr_records(arguments.manifest),
            arguments.checkpoint_dir,
            split=arguments.split,
            device=arguments.device,
            evaluate=evaluate,
        )
        if evaluate and metrics is None:
            raise ValidationError("ATR evaluation did not produce metrics")
        write_jsonl(arguments.output, predictions)
        _write_json(arguments.metrics_output, metrics if evaluate else {
            "status": "not_evaluated", "evaluation_enabled": False,
        })
        return {
            "status": "ok", "predictions": len(predictions),
            "evaluation_status": "evaluated" if evaluate else "not_evaluated",
            "output": arguments.output, "metrics_output": arguments.metrics_output,
        }
    if arguments.command == "atr-synthetic-smoke":
        from .atr import run_synthetic_atr_smoke

        return run_synthetic_atr_smoke(arguments.work_dir, device=arguments.device)
    raise AssertionError(f"unhandled command: {arguments.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(build_parser().parse_args(argv))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
