"""Command-line interface for validation, replay, splits, vocabulary, and metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
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


MAX_SELECTOR_SCORES_FILE_BYTES = 64 * 1024 * 1024
MAX_SELECTOR_SCORE_LINE_BYTES = 64 * 1024
MAX_SELECTOR_MANIFEST_FILE_BYTES = 64 * 1024 * 1024


def _selector_input_signature(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _selector_check_input_stat(info: os.stat_result, budget: int, label: str) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise ValidationError(f"{label} must be a regular file")
    if info.st_size > budget:
        raise ValidationError(f"{label} byte budget exceeded")


def _selector_open_input(path: str, budget: int, label: str):
    """Check path and descriptor budgets before allocating input bytes."""
    try:
        source = Path(path)
        before = source.stat()
        _selector_check_input_stat(before, budget, label)
        descriptor = os.open(source, os.O_RDONLY | os.O_NONBLOCK)
        try:
            opened = os.fstat(descriptor)
            _selector_check_input_stat(opened, budget, label)
            if _selector_input_signature(before) != _selector_input_signature(opened):
                raise ValidationError(f"{label} changed while opening")
            handle = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        return handle, opened
    except OSError as exc:
        raise ValidationError(f"cannot read {label}: {exc}") from exc


def _selector_finish_input(handle, opened: os.stat_result, total: int,
                           budget: int, label: str) -> None:
    after = os.fstat(handle.fileno())
    _selector_check_input_stat(after, budget, label)
    if total != opened.st_size or _selector_input_signature(after) != _selector_input_signature(opened):
        raise ValidationError(f"{label} changed during bounded read")


def _selector_parse_input(raw: bytes, label: str):
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValidationError(f"duplicate {label} JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValidationError(f"nonfinite {label} JSON number: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValidationError(f"nonfinite {label} JSON number")
        return number

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=reject_constant, parse_float=finite_float)
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f"invalid {label} JSON: {exc}") from exc


def _load_selector_score_rows(path: str, max_rows: int) -> list[dict]:
    """Bound each physical line, including newline, before decoding/parse."""
    if type(max_rows) is not int or max_rows < 1:
        raise ValidationError("selector score row budget must be a positive integer")
    label = "selector scores"
    try:
        handle, opened = _selector_open_input(path, MAX_SELECTOR_SCORES_FILE_BYTES, label)
        with handle:
            rows = []
            total = 0
            while True:
                read_limit = min(MAX_SELECTOR_SCORE_LINE_BYTES + 1,
                                 MAX_SELECTOR_SCORES_FILE_BYTES - total + 1)
                raw = handle.readline(read_limit)
                if not raw:
                    break
                total += len(raw)
                if total > MAX_SELECTOR_SCORES_FILE_BYTES:
                    raise ValidationError("selector scores byte budget exceeded")
                if len(raw) > MAX_SELECTOR_SCORE_LINE_BYTES:
                    raise ValidationError("selector score physical line byte budget exceeded")
                if not raw.strip():
                    continue
                if len(rows) >= max_rows:
                    raise ValidationError("selector score row budget exceeded")
                row = _selector_parse_input(raw, "selector score row")
                if type(row) is not dict:
                    raise ValidationError("selector score row must be a JSON object")
                rows.append(row)
            _selector_finish_input(handle, opened, total, MAX_SELECTOR_SCORES_FILE_BYTES, label)
        return rows
    except OSError as exc:
        raise ValidationError(f"cannot read {label}: {exc}") from exc


def _load_selector_manifest(path: str) -> dict:
    """Read only budget plus one sentinel, then check stability and parse."""
    label = "selector manifest"
    try:
        handle, opened = _selector_open_input(path, MAX_SELECTOR_MANIFEST_FILE_BYTES, label)
        with handle:
            raw = handle.read(MAX_SELECTOR_MANIFEST_FILE_BYTES + 1)
            if len(raw) > MAX_SELECTOR_MANIFEST_FILE_BYTES:
                raise ValidationError("selector manifest byte budget exceeded")
            _selector_finish_input(handle, opened, len(raw), MAX_SELECTOR_MANIFEST_FILE_BYTES, label)
        manifest = _selector_parse_input(raw, label)
        if type(manifest) is not dict:
            raise ValidationError("selector manifest must be a JSON object")
        return manifest
    except OSError as exc:
        raise ValidationError(f"cannot read {label}: {exc}") from exc


def _selector_finite_json(value: dict) -> str:
    if not isinstance(value, dict):
        raise ValidationError("selector output must be a JSON object")
    try:
        return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValidationError("selector output must contain finite JSON values") from exc


def _selector_manifest_chunks(value: dict):
    """Yield compact canonical UTF-8 chunks without retaining the full JSON."""
    if not isinstance(value, dict):
        raise ValidationError("selector output must be a JSON object")
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False)
    try:
        for chunk in encoder.iterencode(value):
            yield chunk.encode("utf-8")
        yield b"\n"
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError("selector output must contain finite JSON values encoded as UTF-8") from exc


def _write_new_selector_json(path: str, value: dict) -> None:
    """Preflight a bounded manifest, then stream into an exclusive private file."""
    expected_bytes = 0
    for chunk in _selector_manifest_chunks(value):
        expected_bytes += len(chunk)
        if expected_bytes > MAX_SELECTOR_MANIFEST_FILE_BYTES:
            raise ValidationError("selector manifest byte budget exceeded")
    destination = Path(path)
    missing_parents: list[Path] = []
    parent = destination.parent
    while not parent.exists():
        missing_parents.append(parent)
        parent = parent.parent
    for directory in reversed(missing_parents):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            if not directory.is_dir():
                raise
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    owned = None
    try:
        owned = os.fstat(descriptor)
        # Keep the owned inode open through cleanup even if the stream closes.
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            total = 0
            for chunk in _selector_manifest_chunks(value):
                total += len(chunk)
                if total > MAX_SELECTOR_MANIFEST_FILE_BYTES:
                    raise ValidationError("selector manifest byte budget exceeded")
                if handle.write(chunk) != len(chunk):
                    raise OSError("short selector manifest write")
            if total != expected_bytes:
                raise ValidationError("selector manifest encoding changed between passes")
        try:
            current = destination.lstat()
        except FileNotFoundError as exc:
            raise ValidationError("selector output changed during bounded write") from exc
        actual = os.fstat(descriptor)
        if ((current.st_dev, current.st_ino) != (owned.st_dev, owned.st_ino)
                or current.st_size != expected_bytes or actual.st_size != expected_bytes):
            raise ValidationError("selector output changed during bounded write")
        closing_descriptor, descriptor = descriptor, None
        os.close(closing_descriptor)
    except BaseException:
        if owned is not None:
            try:
                current = destination.lstat()
            except FileNotFoundError:
                pass
            else:
                if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                    destination.unlink()
        raise
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


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

    upstream_generate = commands.add_parser("upstream-generate")
    upstream_generate.add_argument("--config", required=True)
    upstream_generate.add_argument("--output", required=True)

    upstream_reconstruct = commands.add_parser("upstream-reconstruct")
    upstream_reconstruct.add_argument("--input", required=True)
    upstream_reconstruct.add_argument("--config", required=True)
    upstream_reconstruct.add_argument("--output", required=True)

    upstream_smoke = commands.add_parser("upstream-synthetic-smoke")
    upstream_smoke.add_argument("--work-dir", required=True)
    upstream_smoke.add_argument("--device", choices=("cpu",), default="cpu")
    owned_timing = commands.add_parser(
        "owned-gpu-timing-benchmark",
        help="Two owned CUDA workers with exact CPU ground truth and bounded cleanup",
    )
    owned_timing.add_argument("--work-dir", required=True)
    owned_timing.add_argument("--steps", type=int, default=3)
    owned_timing.add_argument("--elements", type=int, default=65536)
    owned_timing.add_argument("--timeout-seconds", type=float, default=20.0)
    selector_build = commands.add_parser("selector-build")
    selector_build.add_argument("--config", required=True)
    selector_build.add_argument("--scores", required=True)
    selector_build.add_argument("--output", required=True)

    selector_validate = commands.add_parser("selector-validate")
    selector_validate.add_argument("--manifest", required=True)
    selector_validate.add_argument("--expected-config", required=True)

    selector_smoke = commands.add_parser("selector-replay-smoke")
    selector_smoke.add_argument("--work-dir", required=True)
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
    if arguments.command == "upstream-generate":
        from .workload import (
            WorkloadConfig, generate_controlled_workload, write_controlled_workload,
        )

        bundle = generate_controlled_workload(WorkloadConfig.from_json(arguments.config))
        write_controlled_workload(bundle, arguments.output)
        counts = {
            split: sum(record["split"] == split for record in bundle["runs"])
            for split in ("train", "validation", "test")
        }
        return {
            "status": "ok", "source_kind": "synthetic_probe_simulation",
            "scientific_result": False, "runs": len(bundle["runs"]),
            "split_counts": counts, "output": arguments.output,
        }
    if arguments.command == "upstream-reconstruct":
        from .probe_contract import load_probe_runs
        from .probe_reconstruction import ReconstructionConfig, reconstruct_probe_run

        records = load_probe_runs(arguments.input)
        config = ReconstructionConfig.from_json(arguments.config)
        profiles = [reconstruct_probe_run(record, config) for record in records]
        _write_json(arguments.output, {
            "schema_version": "janus.probe.reconstruction.batch.v1",
            "scientific_result": False, "profiles": profiles,
        })
        return {
            "status": "ok", "scientific_result": False, "runs": len(profiles),
            "decoding_steps": sum(len(row["decoding_steps"]) for row in profiles),
            "phase_source_kind": "oracle_aligned_phase",
            "output": arguments.output,
        }
    if arguments.command == "upstream-synthetic-smoke":
        from .upstream_bridge import run_upstream_smoke

        return run_upstream_smoke(arguments.work_dir, device=arguments.device)
    if arguments.command == "owned-gpu-timing-benchmark":
        from .owned_gpu_timing import OwnedGPUTimingConfig, run_owned_gpu_timing

        config = OwnedGPUTimingConfig(
            steps=arguments.steps, elements=arguments.elements,
            timeout_seconds=arguments.timeout_seconds,
        )
        return run_owned_gpu_timing(arguments.work_dir, config)
    if arguments.command == "selector-build":
        from .selector_replay import SelectorReplayConfig, build_selector_manifest

        config = SelectorReplayConfig.from_json(arguments.config)
        max_rows = len(config.teacher_forced_token_ids) * config.layers * config.query_heads
        manifest = build_selector_manifest(config, _load_selector_score_rows(arguments.scores, max_rows))
        _write_new_selector_json(arguments.output, manifest)
        return {
            "status": "ok", "manifest_sha256": manifest["manifest_sha256"],
            "steps": len(manifest["steps"]),
            "source_kind": manifest["config"]["source_kind"],
            "scientific_result": False, "output": arguments.output,
        }
    if arguments.command == "selector-validate":
        from .selector_replay import SelectorReplayConfig, validate_selector_manifest

        config = SelectorReplayConfig.from_json(arguments.expected_config)
        manifest = _load_selector_manifest(arguments.manifest)
        validated = validate_selector_manifest(manifest, expected_config=config)
        return {
            "status": "ok", "manifest_sha256": validated["manifest_sha256"],
            "steps": len(validated["steps"]),
            "source_kind": validated["config"]["source_kind"],
            "scientific_result": False,
        }
    if arguments.command == "selector-replay-smoke":
        from .selector_smoke import run_selector_replay_smoke

        report = run_selector_replay_smoke(arguments.work_dir)
        _selector_finite_json(report)
        return report
    raise AssertionError(f"unhandled command: {arguments.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(argv)
        result = run(arguments)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if arguments.command in {"owned-gpu-timing-benchmark", "selector-replay-smoke"} and result.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
