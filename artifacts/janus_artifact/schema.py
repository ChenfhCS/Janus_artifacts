"""JSONL I/O and strict validation for phase-aware trace records."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "janus.trace.v1"
SPLITS = {"unassigned", "train", "validation", "test"}
SOURCE_KINDS = {
    "real_collection",
    "refactored_legacy",
    "oracle_annotation",
    "replay_derived",
    "synthetic_fixture",
}


class ValidationError(ValueError):
    """Raised when an artifact contract is ambiguous or unsafe to evaluate."""


def _require_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field} must be a non-empty string")
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_observations(record: dict[str, Any]) -> None:
    trace = record.get("trace")
    if trace is None:
        if (
            record["source_kind"] == "refactored_legacy"
            and record.get("payload_status") == "lfs_pointer_only"
        ):
            return
        raise ValidationError("trace is required unless a legacy payload is pointer-only")
    if not isinstance(trace, dict):
        raise ValidationError("trace must be an object")
    if trace.get("granularity") not in {"page", "token"}:
        raise ValidationError("trace.granularity must be page or token")
    storage = trace.get("storage", "inline")
    if storage == "external_npz":
        if record["phase"] != "prefill":
            raise ValidationError("the v1 external NPZ adapter supports prefill only")
        if trace.get("data_stage") != "reconstructed_sparsity":
            raise ValidationError("external NPZ must declare reconstructed_sparsity")
        if trace.get("granularity") != "token" or trace.get("aggregation") != "rank_tensor":
            raise ValidationError("external prefill NPZ must be a token-level rank_tensor")
        payload = trace.get("external_payload")
        if not isinstance(payload, dict):
            raise ValidationError("external NPZ requires external_payload metadata")
        digest = payload.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValidationError("external NPZ requires a SHA-256 digest")
        if not isinstance(payload.get("size_bytes"), int) or payload["size_bytes"] <= 0:
            raise ValidationError("external NPZ requires a positive size_bytes")
        arrays = payload.get("arrays")
        if not isinstance(arrays, list) or not arrays:
            raise ValidationError("external NPZ requires array metadata")
        if "observations" in trace:
            raise ValidationError("external NPZ records must not embed observations")
        return
    if storage != "inline":
        raise ValidationError("trace.storage must be inline or external_npz")
    phase = record["phase"]
    expected_aggregation = "cumulative" if phase == "prefill" else "stepwise"
    if trace.get("aggregation") != expected_aggregation:
        raise ValidationError(
            f"{phase} traces require {expected_aggregation} aggregation"
        )
    observations = trace.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValidationError("trace.observations must be a non-empty array")
    if phase == "prefill":
        if not all(_is_number(value) for value in observations):
            raise ValidationError("prefill observations must be a numeric vector")
        return
    widths = set()
    for step in observations:
        if not isinstance(step, list) or not step:
            raise ValidationError("decoding observations must contain non-empty steps")
        if not all(_is_number(value) for value in step):
            raise ValidationError("decoding steps must contain only numbers")
        widths.add(len(step))
    if len(widths) != 1:
        raise ValidationError("decoding observation steps must have equal width")


def validate_trace_record(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise ValidationError("trace record must be an object")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(f"schema_version must equal {SCHEMA_VERSION}")
    _require_string(record, "trace_id")
    _require_string(record, "case_id")
    if record.get("split") not in SPLITS:
        raise ValidationError(f"split must be one of {sorted(SPLITS)}")
    if record.get("phase") not in {"prefill", "decoding"}:
        raise ValidationError("phase must be prefill or decoding")
    if record.get("source_kind") not in SOURCE_KINDS:
        raise ValidationError(f"source_kind must be one of {sorted(SOURCE_KINDS)}")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValidationError("provenance must be an object")

    source_kind = record["source_kind"]
    if source_kind == "real_collection" and not provenance.get("collection_run_id"):
        raise ValidationError("real_collection requires provenance.collection_run_id")
    if source_kind == "refactored_legacy" and not provenance.get("legacy_path"):
        raise ValidationError("refactored_legacy requires provenance.legacy_path")
    if source_kind == "oracle_annotation" and not provenance.get("oracle_method"):
        raise ValidationError("oracle_annotation requires provenance.oracle_method")
    if source_kind == "replay_derived":
        parents = provenance.get("parent_trace_ids")
        if not isinstance(parents, list) or not parents:
            raise ValidationError("replay_derived requires parent_trace_ids")
    if source_kind == "synthetic_fixture" and provenance.get("synthetic") is not True:
        raise ValidationError("synthetic_fixture requires provenance.synthetic=true")

    if record["phase"] == "decoding":
        _require_string(record, "response_id")
    _validate_observations(record)


def validate_group_isolation(
    records: Iterable[dict[str, Any]], group_field: str = "case_id"
) -> None:
    group_splits: dict[str, set[str]] = {}
    for record in records:
        group = record.get(group_field)
        if not isinstance(group, str) or not group:
            raise ValidationError(f"{group_field} must be a non-empty string")
        split = record.get("split")
        if split == "unassigned":
            continue
        group_splits.setdefault(group, set()).add(split)
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked:
        details = ", ".join(
            f"{group}={sorted(splits)}" for group, splits in sorted(leaked.items())
        )
        raise ValidationError(f"group split leakage: {details}")


def validate_trace_records(
    records: Iterable[dict[str, Any]], group_field: str = "case_id"
) -> list[dict[str, Any]]:
    materialized = list(records)
    seen: set[str] = set()
    for record in materialized:
        validate_trace_record(record)
        trace_id = record["trace_id"]
        if trace_id in seen:
            raise ValidationError(f"duplicate trace_id: {trace_id}")
        seen.add(trace_id)
    validate_group_isolation(materialized, group_field=group_field)
    return materialized


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValidationError(f"record at {path}:{line_number} must be an object")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def adapt_legacy_manifest(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt Git LFS pointer metadata without fetching or deserializing payloads."""
    adapted: list[dict[str, Any]] = []
    for entry in entries:
        legacy_path = _require_string(entry, "legacy_path")
        oid = _require_string(entry, "oid_sha256")
        if len(oid) != 64 or any(char not in "0123456789abcdef" for char in oid.lower()):
            raise ValidationError("oid_sha256 must be a 64-character hexadecimal digest")
        size_bytes = entry.get("size_bytes")
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ValidationError("size_bytes must be a positive integer")
        phase = entry.get("phase")
        if phase not in {"prefill", "decoding"}:
            raise ValidationError("manifest phase must be prefill or decoding")
        case_id = _require_string(entry, "case_id")
        digest = hashlib.sha256(legacy_path.encode("utf-8")).hexdigest()[:16]
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "trace_id": f"legacy-pointer-{digest}",
            "case_id": case_id,
            "split": "unassigned",
            "phase": phase,
            "source_kind": "refactored_legacy",
            "payload_status": "lfs_pointer_only",
            "trace": None,
            "labels": {},
            "provenance": {
                "legacy_path": legacy_path,
                "lfs_oid_sha256": oid,
                "lfs_size_bytes": size_bytes,
                "adaptation": "metadata_only_no_deserialization",
            },
        }
        if phase == "decoding":
            record["response_id"] = _require_string(entry, "response_id")
        validate_trace_record(record)
        adapted.append(deepcopy(record))
    return adapted
