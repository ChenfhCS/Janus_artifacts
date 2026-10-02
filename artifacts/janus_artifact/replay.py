"""Deterministic offline replay of already-recorded trace containers."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Iterable

from .schema import SCHEMA_VERSION, ValidationError, validate_trace_record


def replay_record(
    record: dict[str, Any], external_payload_path: str | None = None
) -> dict[str, Any]:
    """Canonicalize one recorded trace; this does not reproduce the side channel."""
    validate_trace_record(record)
    if record.get("trace") is None:
        raise ValidationError(
            f"cannot replay metadata-only legacy pointer: {record.get('trace_id')}"
        )
    if record["trace"].get("storage") == "external_npz":
        if not external_payload_path:
            raise ValidationError("external NPZ replay requires an explicit payload path")
        from .legacy_npz import verify_prefill_rank_npz_record

        verify_prefill_rank_npz_record(record, external_payload_path)
    canonical = json.dumps(record["trace"], sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    replayed: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": f"{record['trace_id']}:replay:{digest[:12]}",
        "case_id": record["case_id"],
        "split": record["split"],
        "phase": record["phase"],
        "source_kind": "replay_derived",
        "trace": deepcopy(record["trace"]),
        "labels": deepcopy(record.get("labels", {})),
        "provenance": {
            "parent_trace_ids": [record["trace_id"]],
            "replay_sha256": digest,
            "operation": "schema_validate_and_canonicalize_only",
            "real_side_channel_reproduction": False,
            "external_payload_verified": record["trace"].get("storage") == "external_npz",
        },
    }
    if record["phase"] == "decoding":
        replayed["response_id"] = record["response_id"]
    validate_trace_record(replayed)
    return replayed


def replay_records(
    records: Iterable[dict[str, Any]], external_payload_path: str | None = None
) -> list[dict[str, Any]]:
    materialized = list(records)
    external_count = sum(
        record.get("trace", {}).get("storage") == "external_npz"
        for record in materialized
        if isinstance(record.get("trace"), dict)
    )
    if external_count > 1 and external_payload_path:
        raise ValidationError("one --external-payload cannot resolve multiple records")
    return [
        replay_record(record, external_payload_path=external_payload_path)
        for record in materialized
    ]
