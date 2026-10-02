"""Evidence-backed manifest for selected legacy prefill rank tensors."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .legacy_npz import inspect_prefill_rank_npz
from .schema import ValidationError


MANIFEST_VERSION = "janus.dataset-manifest.v1"
PAYLOAD_NAME = "q_k_attn_rank_topk256.npz"
REQUIRED_CSV_COLUMNS = ("index", "query", "true_label", "pred_label", "pred_prob")
MAX_CSV_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024

_EVIDENCE_RANGES = (
    (
        "csv_index_to_case_path",
        'case_idx = int(one_s["index"])',
        "self.valid_data.append((one_s, src_npz))",
    ),
    (
        "selected_payload_and_csv_row_share_index",
        'case_idx = item["index"]',
        '"pred_prob": item["pred_prob"]',
    ),
    (
        "output_csv_columns_and_write",
        "out_df = pd.DataFrame",
        "out_df.to_csv",
    ),
    (
        "validation_naming_is_not_a_split_manifest",
        "self.val_csv_path =",
        "self.out_csv_path =",
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: str | Path, *, maximum_bytes: int, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise ValidationError(f"{label} does not exist: {resolved}")
    size = resolved.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValidationError(f"{label} has an invalid or excessive size")
    return resolved


def _parse_legacy_index(legacy_payload_path: str) -> int:
    path = PurePosixPath(legacy_payload_path)
    if path.name != PAYLOAD_NAME:
        raise ValidationError(f"legacy payload name must equal {PAYLOAD_NAME}")
    match = re.fullmatch(r"case_(0|[1-9][0-9]*)", path.parent.name)
    if not match:
        raise ValidationError("legacy payload parent must be named case_<nonnegative index>")
    return int(match.group(1))


def _read_csv_row(csv_path: Path, legacy_index: int) -> tuple[list[str], dict[str, str], int]:
    if csv_path.stat().st_size > MAX_CSV_BYTES:
        raise ValidationError("legacy CSV exceeds the size limit")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if columns is None or len(columns) != len(set(columns)):
            raise ValidationError("legacy CSV must have a unique header")
        missing = [column for column in REQUIRED_CSV_COLUMNS if column not in columns]
        if missing:
            raise ValidationError(f"legacy CSV is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValidationError("legacy CSV is empty")

    by_index: dict[int, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValidationError(f"legacy CSV row {line_number} has excess fields")
        raw_index = row["index"].strip()
        if not re.fullmatch(r"0|[1-9][0-9]*", raw_index):
            raise ValidationError(f"legacy CSV row {line_number} has an invalid index")
        index = int(raw_index)
        if index in by_index:
            raise ValidationError(f"duplicate legacy CSV index: {index}")
        if not row["query"].strip() or not row["true_label"].strip():
            raise ValidationError(f"legacy CSV row {line_number} lacks query or true_label")
        if not row["pred_label"].strip():
            raise ValidationError(f"legacy CSV row {line_number} lacks pred_label")
        try:
            probability = float(row["pred_prob"])
        except ValueError as exc:
            raise ValidationError(
                f"legacy CSV row {line_number} has an invalid pred_prob"
            ) from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValidationError(
                f"legacy CSV row {line_number} pred_prob must be finite and in [0, 1]"
            )
        by_index[index] = row
    if legacy_index not in by_index:
        raise ValidationError(f"legacy CSV has no row for case index {legacy_index}")
    return columns, by_index[legacy_index], len(rows)


def _read_lfs_pointer(pointer_path: Path) -> dict[str, Any]:
    if pointer_path.stat().st_size > 1024:
        raise ValidationError("legacy pointer is not a small Git LFS pointer")
    try:
        lines = pointer_path.read_text(encoding="ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValidationError("legacy pointer must be ASCII text") from exc
    if len(lines) != 3 or lines[0] != "version https://git-lfs.github.com/spec/v1":
        raise ValidationError("legacy pointer has an unsupported Git LFS format")
    oid_match = re.fullmatch(r"oid sha256:([0-9a-f]{64})", lines[1])
    size_match = re.fullmatch(r"size ([1-9][0-9]*)", lines[2])
    if not oid_match or not size_match:
        raise ValidationError("legacy pointer has invalid oid or size metadata")
    return {"oid_sha256": oid_match.group(1), "size_bytes": int(size_match.group(1))}


def _read_source_evidence(script_path: Path, logical_path: str) -> dict[str, Any]:
    if script_path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValidationError("legacy source exceeds the size limit")
    source = script_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    ranges: list[dict[str, Any]] = []
    for claim, start_text, end_text in _EVIDENCE_RANGES:
        starts = [index for index, line in enumerate(lines) if start_text in line]
        ends = [index for index, line in enumerate(lines) if end_text in line]
        candidates = [(start, end) for start in starts for end in ends if end >= start]
        if len(candidates) != 1:
            raise ValidationError(
                f"legacy source does not provide one unambiguous {claim} range"
            )
        start, end = candidates[0]
        excerpt = "\n".join(lines[start : end + 1]) + "\n"
        ranges.append(
            {
                "claim": claim,
                "line_start": start + 1,
                "line_end": end + 1,
                "excerpt_sha256": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "legacy_path": logical_path,
        "sha256": _sha256_file(script_path),
        "evidence_ranges": ranges,
        "inspection": "static_text_only_not_executed",
    }


def build_prefill_dataset_manifest(
    *,
    dataset_id: str,
    csv_path: str | Path,
    csv_legacy_path: str,
    script_path: str | Path,
    script_legacy_path: str,
    pointer_path: str | Path,
    payload_path: str | Path,
    payload_legacy_path: str,
) -> dict[str, Any]:
    """Join one verified materialized payload to one uniquely indexed CSV row."""
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValidationError("dataset_id must be a non-empty string")
    for name, value in (
        ("csv_legacy_path", csv_legacy_path),
        ("script_legacy_path", script_legacy_path),
        ("payload_legacy_path", payload_legacy_path),
    ):
        if not isinstance(value, str) or not value:
            raise ValidationError(f"{name} must be a non-empty string")

    csv_file = _require_file(csv_path, maximum_bytes=MAX_CSV_BYTES, label="legacy CSV")
    source_file = _require_file(
        script_path, maximum_bytes=MAX_SOURCE_BYTES, label="legacy source"
    )
    pointer_file = _require_file(pointer_path, maximum_bytes=1024, label="Git LFS pointer")
    payload_file = _require_file(
        payload_path, maximum_bytes=128 * 1024 * 1024, label="materialized NPZ"
    )

    legacy_index = _parse_legacy_index(payload_legacy_path)
    columns, row, csv_row_count = _read_csv_row(csv_file, legacy_index)
    pointer = _read_lfs_pointer(pointer_file)
    payload = inspect_prefill_rank_npz(payload_file)
    if pointer["oid_sha256"] != payload["sha256"]:
        raise ValidationError("materialized NPZ SHA-256 does not match the Git LFS pointer")
    if pointer["size_bytes"] != payload["size_bytes"]:
        raise ValidationError("materialized NPZ size does not match the Git LFS pointer")
    source_evidence = _read_source_evidence(source_file, script_legacy_path)

    sample_digest = hashlib.sha256(
        f"{dataset_id}\0{legacy_index}".encode("utf-8")
    ).hexdigest()[:20]
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_id": dataset_id,
        "scope": "one_selected_legacy_prefill_sample",
        "sources": {
            "csv": {
                "legacy_path": csv_legacy_path,
                "sha256": _sha256_file(csv_file),
                "columns": columns,
                "row_count": csv_row_count,
            },
            "legacy_source": source_evidence,
        },
        "join_contract": {
            "csv_key": "index",
            "case_directory_template": "case_{index}",
            "payload_name": PAYLOAD_NAME,
            "cardinality": "exactly_one_csv_row_per_included_payload",
            "authority": "statically_inspected_legacy_source",
        },
        "split_contract": {
            "split": "unassigned",
            "legacy_naming_observed": "validation",
            "reason_unassigned": (
                "the CSV has no split column and no versioned split manifest was found; "
                "legacy variable and directory names are not promoted to a canonical split"
            ),
        },
        "entries": [
            {
                "manifest_sample_id": f"legacy-sample-{sample_digest}",
                "id_origin": "derived_by_manifest_from_dataset_id_and_legacy_index",
                "legacy_index": legacy_index,
                "case_id": f"case-{legacy_index}",
                "split": "unassigned",
                "query": {
                    "source_column": "query",
                    "present": True,
                    "sha256_utf8": hashlib.sha256(
                        row["query"].encode("utf-8")
                    ).hexdigest(),
                    "text_embedded": False,
                },
                "labels": {
                    "attribute": {
                        "value": row["true_label"],
                        "source_column": "true_label",
                    },
                    "tokens": None,
                },
                "legacy_selected_prediction": {
                    "value": row["pred_label"],
                    "probability": float(row["pred_prob"]),
                    "source_columns": ["pred_label", "pred_prob"],
                    "controlled_rerun": False,
                },
                "payload": {
                    "legacy_path": payload_legacy_path,
                    "git_lfs_pointer": pointer,
                    "materialized_inspection": payload,
                    "pointer_matches_materialized_payload": True,
                },
                "evaluation_eligibility": {
                    "pasr": False,
                    "dasr": False,
                    "reasons": [
                        "canonical split is unassigned",
                        "the existing prediction is selected legacy output, not a controlled rerun",
                        "token labels and token alignment are absent",
                        "the payload is a reconstructed rank tensor, not a raw probe trace",
                    ],
                },
                "missing_fields": [
                    "stable_original_sample_id",
                    "canonical_split_assignment",
                    "token_labels",
                    "token_alignment_contract",
                    "controlled_prediction_by_manifest_sample_id",
                    "raw_probe_trace",
                ],
            }
        ],
    }
    validate_prefill_dataset_manifest(manifest)
    return manifest


def validate_prefill_dataset_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict) or manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValidationError(f"manifest_version must equal {MANIFEST_VERSION}")
    if not isinstance(manifest.get("dataset_id"), str) or not manifest["dataset_id"]:
        raise ValidationError("dataset_id must be a non-empty string")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValidationError("manifest sources must be an object")
    csv_source = sources.get("csv")
    source_code = sources.get("legacy_source")
    if not isinstance(csv_source, dict) or not isinstance(source_code, dict):
        raise ValidationError("manifest must describe CSV and legacy source evidence")
    for source in (csv_source, source_code):
        digest = source.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValidationError("manifest source SHA-256 is invalid")
    evidence = source_code.get("evidence_ranges")
    expected_claims = {item[0] for item in _EVIDENCE_RANGES}
    if not isinstance(evidence, list) or {item.get("claim") for item in evidence} != expected_claims:
        raise ValidationError("manifest source evidence claims are incomplete")
    for item in evidence:
        if (
            not isinstance(item.get("line_start"), int)
            or not isinstance(item.get("line_end"), int)
            or item["line_start"] <= 0
            or item["line_end"] < item["line_start"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("excerpt_sha256", "")))
        ):
            raise ValidationError("manifest source evidence range is invalid")

    join = manifest.get("join_contract")
    if not isinstance(join, dict) or (
        join.get("csv_key"),
        join.get("case_directory_template"),
        join.get("payload_name"),
    ) != ("index", "case_{index}", PAYLOAD_NAME):
        raise ValidationError("manifest join contract is invalid")
    split = manifest.get("split_contract")
    if not isinstance(split, dict) or split.get("split") != "unassigned":
        raise ValidationError("manifest split must remain unassigned without a split manifest")

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValidationError("manifest entries must be a non-empty array")
    sample_ids: set[str] = set()
    legacy_indexes: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValidationError("manifest entry must be an object")
        sample_id = entry.get("manifest_sample_id")
        index = entry.get("legacy_index")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValidationError("manifest_sample_id must be a non-empty string")
        if sample_id in sample_ids:
            raise ValidationError(f"duplicate manifest_sample_id: {sample_id}")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValidationError("legacy_index must be a nonnegative integer")
        if index in legacy_indexes:
            raise ValidationError(f"duplicate legacy_index: {index}")
        sample_ids.add(sample_id)
        legacy_indexes.add(index)
        if entry.get("case_id") != f"case-{index}" or entry.get("split") != "unassigned":
            raise ValidationError("manifest case_id or split does not match its legacy index")
        query = entry.get("query")
        if (
            not isinstance(query, dict)
            or query.get("text_embedded") is not False
            or not re.fullmatch(r"[0-9a-f]{64}", str(query.get("sha256_utf8", "")))
        ):
            raise ValidationError("manifest query metadata is invalid")
        labels = entry.get("labels")
        if (
            not isinstance(labels, dict)
            or labels.get("tokens") is not None
            or not isinstance(labels.get("attribute"), dict)
            or not isinstance(labels["attribute"].get("value"), str)
            or not labels["attribute"]["value"]
        ):
            raise ValidationError("manifest labels must contain only an attribute label")
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            raise ValidationError("manifest payload metadata is required")
        if _parse_legacy_index(str(payload.get("legacy_path", ""))) != index:
            raise ValidationError("manifest payload path does not match legacy_index")
        pointer = payload.get("git_lfs_pointer")
        materialized = payload.get("materialized_inspection")
        if not isinstance(pointer, dict) or not isinstance(materialized, dict):
            raise ValidationError("manifest pointer and materialized metadata are required")
        if (
            pointer.get("oid_sha256") != materialized.get("sha256")
            or pointer.get("size_bytes") != materialized.get("size_bytes")
            or payload.get("pointer_matches_materialized_payload") is not True
        ):
            raise ValidationError("manifest Git LFS pointer does not match materialized payload")
        eligibility = entry.get("evaluation_eligibility")
        if (
            not isinstance(eligibility, dict)
            or eligibility.get("pasr") is not False
            or eligibility.get("dasr") is not False
        ):
            raise ValidationError("selected legacy sample must remain evaluation-ineligible")


def verify_prefill_dataset_manifest(
    manifest: dict[str, Any],
    *,
    csv_path: str | Path,
    script_path: str | Path,
    pointer_path: str | Path,
    payload_path: str | Path,
) -> None:
    """Rebuild a manifest from source bytes and require exact equality."""
    validate_prefill_dataset_manifest(manifest)
    sources = manifest["sources"]
    entry = manifest["entries"][0]
    rebuilt = build_prefill_dataset_manifest(
        dataset_id=manifest["dataset_id"],
        csv_path=csv_path,
        csv_legacy_path=sources["csv"]["legacy_path"],
        script_path=script_path,
        script_legacy_path=sources["legacy_source"]["legacy_path"],
        pointer_path=pointer_path,
        payload_path=payload_path,
        payload_legacy_path=entry["payload"]["legacy_path"],
    )
    if json.dumps(rebuilt, sort_keys=True) != json.dumps(manifest, sort_keys=True):
        raise ValidationError("manifest does not match the supplied source bytes")
