"""Synthetic controlled probes to replay and existing offline model pipelines.

Oracle phase/gold annotations remain separate from raw probes. The QAI integer
rank surrogate is a declared lossy adapter, not original attention ranks. No
physical collection or paper reproduction is performed by this bridge.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .atr import load_atr_checkpoint, predict_atr, train_atr
from .atr_data import ATR_SCHEMA_VERSION, freeze_atr_vocabulary, load_atr_records, validate_atr_records
from .atr_features import ATRConfig
from .legacy_npz import inspect_prefill_rank_npz
from .probe_contract import canonical_sha256
from .probe_reconstruction import ReconstructionConfig, reconstruct_probe_run
from .qai import QAIConfig, QAIRecord, freeze_qai_labels, load_qai_checkpoint, load_qai_records, predict_qai, train_qai
from .replay import replay_records
from .schema import SCHEMA_VERSION, ValidationError, validate_trace_records, write_jsonl
from .workload import WorkloadConfig, generate_controlled_workload, validate_controlled_workload, write_controlled_workload

BRIDGE_VERSION = "janus.controlled_upstream_bridge.v1"
QAI_ADAPTER_VERSION = "janus.synthetic_prefill_rank_surrogate.v1"
QAI_QUANTIZATION_MAX = 65535
QAI_SELECTED_RANKS = 2


def _write_json(path: Path, value: Any) -> None:
    canonical_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _fresh_destination(path: Path) -> None:
    if path.exists():
        raise ValidationError("upstream output directory must be new; existing files are preserved")


def _qai_rank_surrogate(profile: Any) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 3 or any(size <= 0 for size in values.shape) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValidationError("synthetic QAI adapter requires finite nonnegative layer/head/token profiles")
    maximum = float(values.max())
    normalized = values / maximum if maximum else np.zeros_like(values)
    ranks = np.rint(normalized * QAI_QUANTIZATION_MAX).astype(np.uint16)[:, :, None, :]
    return ranks, {
        "adapter_version": QAI_ADAPTER_VERSION, "source_kind": "synthetic_fixture",
        "input_stage": "reconstructed_token_sparsity", "output_stage": "synthetic_quantized_rank_surrogate",
        "normalization": "divide_by_this_run_global_max_zero_if_all_zero",
        "normalization_fit_scope": "per_run_no_dataset_statistics", "input_maximum": maximum,
        "quantization": {"dtype": "uint16", "maximum": QAI_QUANTIZATION_MAX, "rounding": "nearest_even"},
        "query_axis": "one_synthetic_query_axis", "top_k": QAI_SELECTED_RANKS,
        "qai_selected_ranks": QAI_SELECTED_RANKS,
        "selection": "existing_qai_largest_values_histogram_then_per_layer_head_minmax",
        "information_loss": [
            "global magnitude removed", "float values quantized to integers",
            "original query axis unavailable; a singleton is used",
            "QAI histogram retains selected membership rather than numeric amplitudes",
        ],
        "original_attention_ranks": False, "scientific_result": False,
    }


def _trace_records(run: dict[str, Any], reconstruction: dict[str, Any]) -> list[dict[str, Any]]:
    events = reconstruction["page_events"]
    page_order = [row["page_id"] for row in events["prefill"]]
    if any([row["page_id"] for row in step["pages"]] != page_order for step in events["decoding"]):
        raise ValidationError("reconstructed page observation order changes between phases or steps")
    numeric = {
        "page": (
            [row["relative_frequency_proxy"] for row in events["prefill"]],
            [[row["event"] for row in step["pages"]] for step in events["decoding"]],
        ),
        "token": (
            np.asarray(reconstruction["prefill_profile"], dtype=np.float64).reshape(-1).tolist(),
            [np.asarray(step["profile"], dtype=np.float64).reshape(-1).tolist() for step in reconstruction["decoding_steps"]],
        ),
    }
    records = []
    for granularity, (prefill, decoding) in numeric.items():
        for phase, observations in (("prefill", prefill), ("decoding", decoding)):
            record = {
                "schema_version": SCHEMA_VERSION,
                "trace_id": f"{run['run_id']}:{phase}:{granularity}:reconstructed",
                "case_id": run["case_id"], "split": run["split"], "phase": phase,
                "source_kind": "synthetic_fixture",
                "trace": {
                    "storage": "inline", "granularity": granularity,
                    "aggregation": "cumulative" if phase == "prefill" else "stepwise",
                    "data_stage": f"reconstructed_{granularity}_sparsity",
                    "phase_source_kind": "oracle_aligned_phase", "observations": observations,
                    "axis_order": ["layer", "kv_head", "key_token"] if granularity == "token" else ["declared_page_order"],
                    "ordered_layer_ids": run["layout"]["layer_ids"],
                    "ordered_kv_head_ids": run["layout"]["kv_head_ids"], "token_width": run["layout"]["token_width"],
                    "ordered_page_ids": page_order if granularity == "page" else [],
                    "step_ids": [step["step_id"] for step in reconstruction["decoding_steps"]] if phase == "decoding" else [],
                    "prefill_values": "relative_frequency_proxy_not_exact_access_counts",
                },
                "labels": {"source_kind": "oracle_annotation", "attributes": run["gold"]["attributes"]} if phase == "prefill" else {
                    "source_kind": "oracle_annotation", "gold_steps": run["gold"]["steps"],
                    "tokenizer": run["gold"]["tokenizer"], "alignment": run["gold"]["alignment"],
                },
                "provenance": {
                    "synthetic": True, "bridge_version": BRIDGE_VERSION, "parent_probe_run_ids": [run["run_id"]],
                    "input_source_kind": run["source_kind"], "input_sha256": canonical_sha256(run),
                    "reconstruction": reconstruction["provenance"], "phase_source_kind": "oracle_aligned_phase",
                    "alignment_evidence_id": run["phase_reference"]["alignment_evidence_id"],
                    "split_assignment_id": run["split_assignment_id"], "allocation_epoch": run["allocation_epoch"],
                    "scientific_result": False,
                },
            }
            if phase == "decoding":
                record["response_id"] = run["response_id"]
            records.append(record)
    return records


def _atr_record(run: dict[str, Any], reconstruction: dict[str, Any], config: ReconstructionConfig) -> dict[str, Any]:
    profiles = {(step["step_id"], step["step_index"]): step["profile"] for step in reconstruction["decoding_steps"]}
    gold_steps = run["gold"]["steps"]
    if set(profiles) != {(step["step_id"], step["step_index"]) for step in gold_steps}:
        raise ValidationError("reconstructed steps do not match explicit oracle step mapping")
    return {
        "schema_version": ATR_SCHEMA_VERSION,
        "response_id": run["response_id"], "case_id": run["case_id"], "task_id": run["task_id"], "split": run["split"],
        "source_kind": "synthetic_fixture",
        "provenance": {
            "synthetic": True, "synthetic_simulation_run_id": run["run_id"],
            "probe_input_sha256": canonical_sha256(run),
            "profile_source_sha256": canonical_sha256(reconstruction["decoding_steps"]),
            "split_assignment_id": run["split_assignment_id"],
            "alignment_evidence_id": run["phase_reference"]["alignment_evidence_id"],
            "phase_source_kind": "oracle_aligned_phase", "bridge_version": BRIDGE_VERSION,
        },
        "tokenizer": run["gold"]["tokenizer"], "alignment": run["gold"]["alignment"],
        "feature_contract": {
            "data_stage": "reconstructed_token_sparsity", "axis_order": ["layer", "kv_head", "key_token"],
            "layer_ids": run["layout"]["layer_ids"], "kv_head_ids": run["layout"]["kv_head_ids"],
            "key_position_policy": "absolute_zero_based_prefix_positions",
            "reconstruction": {
                "method": "controlled_probe_numeric_reconstruction",
                "revision": reconstruction["provenance"]["reconstruction_revision"], "parameters": asdict(config),
            },
        },
        "steps": [{
            "step_id": step["step_id"], "step_index": step["step_index"], "gold_token_id": step["gold_token_id"],
            "profile": profiles[(step["step_id"], step["step_index"])],
        } for step in gold_steps],
    }


def build_controlled_artifacts(
    bundle: dict[str, Any], config: ReconstructionConfig, output_dir: str | Path,
) -> dict[str, Any]:
    """Validate a complete synthetic bundle before creating any output.

    Offline reconstruction of physical recordings is an upstream numerical API;
    it is outside this synthetic QAI/ATR adapter boundary.
    """
    bundle = validate_controlled_workload(bundle)
    if any(run["source_kind"] != "synthetic_probe_simulation" for run in bundle["runs"]):
        raise ValidationError("controlled artifact bridge accepts synthetic simulations only")
    if not isinstance(config, ReconstructionConfig):
        raise ValidationError("bridge config must be ReconstructionConfig")
    config.validate()
    destination = Path(output_dir)
    _fresh_destination(destination)
    reconstructions = [reconstruct_probe_run(run, config) for run in bundle["runs"]]
    traces: list[dict[str, Any]] = []
    atr_records, qai_arrays, adapter_provenance, mapping = [], [], [], []
    rank_splits: dict[str, str] = {}
    for run, reconstruction in zip(bundle["runs"], reconstructions):
        if any(reconstruction[key] != run[key] for key in ("run_id", "sample_id", "case_id", "response_id", "split")):
            raise ValidationError("reconstruction changed a frozen workload identity or split")
        traces.extend(_trace_records(run, reconstruction))
        atr_records.append(_atr_record(run, reconstruction, config))
        ranks, adapter = _qai_rank_surrogate(reconstruction["prefill_profile"])
        # Compare numeric payloads before ZIP serialization. IDs, filenames, ZIP
        # timestamps and watermarks cannot distinguish identical numeric inputs.
        array_digest = canonical_sha256({"shape": list(ranks.shape), "values": ranks.tolist(), "top_k": QAI_SELECTED_RANKS})
        prior_split = rank_splits.get(array_digest)
        if prior_split is not None and prior_split != run["split"]:
            raise ValidationError("synthetic QAI rank-surrogate payload split leakage; vary actual toy access schedule")
        rank_splits[array_digest] = run["split"]
        qai_arrays.append(ranks)
        adapter_provenance.append({
            "sample_id": run["sample_id"], "case_id": run["case_id"], "split": run["split"],
            "input_sha256": canonical_sha256(run), "rank_payload_sha256": array_digest, "adapter": adapter,
        })
        mapping.append({
            "run_id": run["run_id"], "sample_id": run["sample_id"], "case_id": run["case_id"],
            "response_id": run["response_id"], "split": run["split"], "split_assignment_id": run["split_assignment_id"],
            "input_sha256": canonical_sha256(run), "reconstruction_sha256": canonical_sha256(reconstruction),
            "phase_source_kind": "oracle_aligned_phase", "gold_source_kind": "oracle_annotation",
            "tokenizer": run["gold"]["tokenizer"], "alignment": run["gold"]["alignment"],
            "step_mapping": [{key: step[key] for key in ("step_id", "step_index", "gold_token_id")} for step in run["gold"]["steps"]],
        })
    validate_trace_records(traces)
    replayed = replay_records(traces)
    validate_trace_records(replayed)
    atr_records = validate_atr_records(atr_records, require_gold=True)
    vocabulary = freeze_atr_vocabulary(atr_records)
    # Validate the existing classifier vocabulary without reading or inventing
    # a payload. Oracle sidecar corrections may remove a train class.
    freeze_qai_labels([QAIRecord(
        sample_id=run["sample_id"], case_id=run["case_id"], split=run["split"],
        attribute="toy_topic", attribute_present=True,
        label=run["gold"]["attributes"]["toy_topic"],
        npz_path="not-loaded-during-label-validation.npz",
    ) for run in bundle["runs"]])
    destination.mkdir(parents=True, exist_ok=False)
    write_controlled_workload(bundle, destination / "controlled_workload.json")
    write_jsonl(destination / "probe_runs.jsonl", bundle["runs"])
    write_jsonl(destination / "reconstructed_profiles.jsonl", reconstructions)
    write_jsonl(destination / "traces.jsonl", traces)
    write_jsonl(destination / "replayed_traces.jsonl", replayed)
    qai_manifest = []
    for run, ranks in zip(bundle["runs"], qai_arrays):
        filename = f"sample-{canonical_sha256({'sample_id': run['sample_id']})[:24]}.npz"
        payload = destination / "qai" / "payloads" / filename
        payload.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(payload, attn_rank=ranks, top_k=np.asarray(QAI_SELECTED_RANKS, dtype=np.uint16))
        inspect_prefill_rank_npz(payload)
        qai_manifest.append({
            "sample_id": run["sample_id"], "case_id": run["case_id"], "split": run["split"],
            "attribute": "toy_topic", "attribute_present": True,
            "label": run["gold"]["attributes"]["toy_topic"], "npz_path": f"payloads/{filename}",
        })
    qai_path, atr_path = destination / "qai" / "manifest.jsonl", destination / "atr" / "manifest.jsonl"
    write_jsonl(qai_path, qai_manifest)
    load_qai_records(qai_path)  # The existing byte-digest split guard also runs.
    write_jsonl(atr_path, atr_records)
    load_atr_records(atr_path)
    _write_json(destination / "qai" / "adapter_provenance.json", adapter_provenance)
    _write_json(destination / "mapping_provenance.json", mapping)
    _write_json(destination / "atr" / "candidate_vocabulary.json", vocabulary)
    report = {
        "bridge_version": BRIDGE_VERSION, "status": "ok", "synthetic_only": True,
        "scientific_result": False, "physical_collection_performed": False,
        "input_bundle_sha256": canonical_sha256(bundle), "reconstruction_config": asdict(config),
        "phase_source_kind": "oracle_aligned_phase", "split_contract": bundle["split_contract"],
        "tokenizer": bundle["tokenizer"],
        "split_response_counts": dict(sorted(Counter(run["split"] for run in bundle["runs"]).items())),
        "probe_runs": len(bundle["runs"]), "trace_records": len(traces), "replayed_trace_records": len(replayed),
        "qai_records": len(qai_manifest), "atr_responses": len(atr_records),
        "atr_steps": sum(len(record["steps"]) for record in atr_records),
        "candidate_token_ids_train_only": vocabulary["token_ids"], "qai_adapter": adapter_provenance[0]["adapter"],
        "paths": {name: str(destination / relative) for name, relative in {
            "controlled_workload": "controlled_workload.json", "probe_runs": "probe_runs.jsonl",
            "reconstructed_profiles": "reconstructed_profiles.jsonl", "traces": "traces.jsonl",
            "replayed_traces": "replayed_traces.jsonl", "qai_manifest": "qai/manifest.jsonl",
            "qai_adapter_provenance": "qai/adapter_provenance.json", "atr_manifest": "atr/manifest.jsonl",
            "atr_candidate_vocabulary": "atr/candidate_vocabulary.json", "mapping_provenance": "mapping_provenance.json",
        }.items()},
    }
    _write_json(destination / "build_report.json", report)
    return report


def run_upstream_smoke(output_dir: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    """Run actual tiny gradient/save-load/inference checks on known synthetic data."""
    if device != "cpu":
        raise ValidationError("controlled upstream smoke supports the explicitly authorized CPU device only")
    destination = Path(output_dir)
    _fresh_destination(destination)
    started = time.monotonic()
    bundle = generate_controlled_workload(WorkloadConfig())
    build = build_controlled_artifacts(bundle, ReconstructionConfig(), destination / "artifacts")
    qai_records = load_qai_records(build["paths"]["qai_manifest"])
    atr_records = load_atr_records(build["paths"]["atr_manifest"])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        qai_training = train_qai(
            qai_records, QAIConfig(selected_ranks=QAI_SELECTED_RANKS, base_channels=2, batch_size=4, epochs=1, seed=19),
            destination / "qai_checkpoint", device="cpu", run_kind="synthetic_smoke_only",
        )
        qai_model, _, qai_device = load_qai_checkpoint(destination / "qai_checkpoint", device="cpu")
        qai_predictions, pasr = predict_qai(qai_records, destination / "qai_checkpoint", split="test", device="cpu")
        atr_training = train_atr(
            atr_records, ATRConfig(base_channels=2, batch_size=4, epochs=1, seed=19),
            destination / "atr_checkpoint", device="cpu", run_kind="synthetic_smoke_only",
        )
        atr_model, _, atr_device = load_atr_checkpoint(destination / "atr_checkpoint", device="cpu")
        atr_predictions, dasr = predict_atr(atr_records, destination / "atr_checkpoint", device="cpu")
    finally:
        torch.set_num_threads(previous_threads)
    test_records = [record for record in atr_records if record["split"] == "test"]
    gold_tokens = [step["gold_token_id"] for record in test_records for step in record["steps"]]
    candidates = set(build["candidate_token_ids_train_only"])
    if (
        qai_model.training or atr_model.training or str(qai_device) != "cpu" or str(atr_device) != "cpu"
        or not qai_training["gradient_observed"] or not qai_training["parameter_changed"]
        or not atr_training["gradient_observed"] or not atr_training["parameter_changed"]
        or pasr["micro_denominator_present_queries"] != len(test_records)
        or dasr is None or len(atr_predictions) != len(gold_tokens)
        or dasr["micro_denominator_all_gold_tokens"] != len(gold_tokens)
        or dasr["out_of_vocabulary_gold_tokens_counted_incorrect"] != sum(token not in candidates for token in gold_tokens)
    ):
        raise ValidationError("controlled upstream smoke failed gradient/reload/full-denominator invariants")
    write_jsonl(destination / "qai_predictions.jsonl", qai_predictions)
    write_jsonl(destination / "atr_predictions.jsonl", atr_predictions)
    report = {
        "status": "ok", "synthetic_only": True, "scientific_result": False,
        "physical_collection_performed": False, "paper_reproduction": False,
        "duration_seconds": time.monotonic() - started, "build": build,
        "qai_training": qai_training, "atr_training": atr_training,
        "qai_checkpoint_reload_verified": True, "atr_checkpoint_reload_verified": True,
        "checkpoint_loading": "existing_loaders_with_weights_only_true",
        "pasr_smoke_only": pasr, "dasr_smoke_only": dasr,
        "qai_test_predictions": len(qai_predictions), "atr_test_predictions": len(atr_predictions),
    }
    _write_json(destination / "report.json", report)
    return report
