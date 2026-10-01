"""Strict response/step contracts for reconstructed ATR token profiles.

This is a data boundary for software validation, not a collector. Token alignment
and reconstruction metadata must be supplied explicitly. Digests detect changes
and accidental split leakage; they do not authenticate a malicious producer.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .schema import ValidationError
from .splits import assign_grouped_splits

ATR_SCHEMA_VERSION = "janus.atr.response.v1"
ATR_VOCABULARY_VERSION = "janus.atr.vocabulary.v1"
ATR_SPLITS = {"train", "validation", "test"}
MAX_ATR_FILE_BYTES = 64 * 1024 * 1024
MAX_ATR_PROFILE_ELEMENTS = 8_000_000
_F32 = struct.Struct("<f")
_RECORD_FIELDS = {
    "schema_version", "response_id", "case_id", "task_id", "split",
    "source_kind", "provenance", "tokenizer", "alignment", "feature_contract", "steps",
}
_CONTRACT_FIELDS = {"schema_version", "task_id", "tokenizer", "alignment", "feature_contract"}
_VOCABULARY_FIELDS = {
    "schema_version", "task_id", "tokenizer", "source_split", "frozen", "token_ids", "sha256",
}


def _fields(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValidationError(f"{name} fields must equal {sorted(expected)}")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValidationError(f"{name} must be valid UTF-8 text") from exc
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _primitive(value: Any, name: str, depth: int = 0) -> None:
    if depth > 32:
        raise ValidationError(f"{name} exceeds the metadata nesting budget")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValidationError(f"{name} must contain valid UTF-8 text") from exc
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError):
            finite = False
        if not finite:
            raise ValidationError(f"{name} must contain finite JSON numbers")
        return
    if isinstance(value, list):
        for item in value:
            _primitive(item, name, depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _primitive(item, name, depth + 1)
        return
    raise ValidationError(f"{name} must contain primitive JSON values")


def _digest(value: Any) -> str:
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    try:
        for chunk in encoder.iterencode(value):
            digest.update(chunk.encode("utf-8"))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValidationError("ATR digest input must contain finite primitive UTF-8 JSON values") from exc
    return digest.hexdigest()


def _tokenizer(value: Any) -> dict[str, Any]:
    tokenizer = _fields(value, {"name", "revision", "vocab_size"}, "tokenizer")
    _string(tokenizer["name"], "tokenizer.name")
    _string(tokenizer["revision"], "tokenizer.revision")
    _integer(tokenizer["vocab_size"], "tokenizer.vocab_size", 1)
    return tokenizer


def _ids(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{name} must be a non-empty array")
    for item in value:
        _string(item, name)
    if len(set(value)) != len(value):
        raise ValidationError(f"{name} must contain unique strings")
    return value


def _validate_contract(record: dict[str, Any]) -> None:
    _tokenizer(record["tokenizer"])
    alignment = _fields(record["alignment"], {
        "step_index_base", "profile_predicts", "bos_included", "eos_included",
        "special_tokens", "response_scope",
    }, "alignment")
    if _integer(alignment["step_index_base"], "alignment.step_index_base") != 0:
        raise ValidationError("alignment.step_index_base must be zero")
    if alignment["profile_predicts"] != "same_index_output_token":
        raise ValidationError("alignment.profile_predicts must equal same_index_output_token")
    if not isinstance(alignment["bos_included"], bool) or not isinstance(alignment["eos_included"], bool):
        raise ValidationError("alignment BOS/EOS inclusion must be boolean")
    _string(alignment["special_tokens"], "alignment.special_tokens")
    if alignment["special_tokens"] not in {"included", "excluded"}:
        raise ValidationError("alignment.special_tokens must be included or excluded")
    if alignment["special_tokens"] == "excluded" and (alignment["bos_included"] or alignment["eos_included"]):
        raise ValidationError("alignment excludes special tokens but declares BOS/EOS included")
    if alignment["response_scope"] != "complete_response":
        raise ValidationError("alignment.response_scope must equal complete_response")
    feature = _fields(record["feature_contract"], {
        "data_stage", "axis_order", "layer_ids", "kv_head_ids",
        "key_position_policy", "reconstruction",
    }, "feature_contract")
    if feature["data_stage"] != "reconstructed_token_sparsity":
        raise ValidationError("feature_contract.data_stage must equal reconstructed_token_sparsity")
    if feature["axis_order"] != ["layer", "kv_head", "key_token"]:
        raise ValidationError("feature_contract.axis_order must equal [layer, kv_head, key_token]")
    _ids(feature["layer_ids"], "feature_contract.layer_ids")
    _ids(feature["kv_head_ids"], "feature_contract.kv_head_ids")
    if feature["key_position_policy"] != "absolute_zero_based_prefix_positions":
        raise ValidationError("feature_contract.key_position_policy must equal absolute_zero_based_prefix_positions")
    reconstruction = _fields(feature["reconstruction"], {"method", "revision", "parameters"}, "reconstruction")
    _string(reconstruction["method"], "reconstruction.method")
    _string(reconstruction["revision"], "reconstruction.revision")
    if not isinstance(reconstruction["parameters"], dict):
        raise ValidationError("reconstruction.parameters must be a primitive JSON object")
    _primitive(reconstruction["parameters"], "reconstruction.parameters")


def _float32_bytes(value: int | float) -> bytes:
    """Canonical model input bytes: little endian float32 and positive zero."""
    try:
        packed = _F32.pack(value)
        converted = _F32.unpack(packed)[0]
    except (OverflowError, ValueError, TypeError, struct.error) as exc:
        raise ValidationError("profile value must remain finite after float32 conversion") from exc
    if not math.isfinite(converted):
        raise ValidationError("profile value must remain finite after float32 conversion")
    return b"\x00\x00\x00\x00" if converted == 0 else packed


def _profile_digest(profiles: list[Any]) -> str | None:
    """Hash semantic model input, ignoring IDs, gold and JSON number spelling.

    A versioned stream contains the step count, observed/missing flags, observed
    shapes, and model-dtype float32 values. Small blocks bound temporary storage.
    This is integrity/leakage detection, not producer authentication.
    """
    if not any(profile is not None for profile in profiles):
        return None
    digest = hashlib.sha256(b"janus.atr.profile-f32le.v1\x00")
    digest.update(struct.pack("<Q", len(profiles)))
    block = bytearray()
    for profile in profiles:
        digest.update(b"\x00" if profile is None else b"\x01")
        if profile is None:
            continue
        digest.update(struct.pack("<QQQ", len(profile), len(profile[0]), len(profile[0][0])))
        for layer in profile:
            for head in layer:
                for value in head:
                    block.extend(_float32_bytes(value))
                    if len(block) >= 65536:
                        digest.update(block)
                        block.clear()
        if block:
            digest.update(block)
            block.clear()
    return digest.hexdigest()


def _validate_profile(profile: Any, record: dict[str, Any], used: int) -> int:
    feature = record["feature_contract"]
    if not isinstance(profile, list) or len(profile) != len(feature["layer_ids"]):
        raise ValidationError("profile layer dimension does not match feature_contract")
    width: int | None = None
    for layer in profile:
        if not isinstance(layer, list) or len(layer) != len(feature["kv_head_ids"]):
            raise ValidationError("profile kv_head dimension does not match feature_contract")
        for head in layer:
            if not isinstance(head, list) or not head:
                raise ValidationError("profile key_token vectors must be non-empty arrays")
            if width is None:
                width = len(head)
            if len(head) != width:
                raise ValidationError("profile key_token width must agree across layer/head axes within each step")
            used += len(head)
            if used > MAX_ATR_PROFILE_ELEMENTS:
                raise ValidationError("ATR profile element budget exceeded")
            for value in head:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValidationError("profile must contain finite nonnegative numbers")
                try:
                    valid = math.isfinite(value) and value >= 0
                except (OverflowError, TypeError):
                    valid = False
                if not valid:
                    raise ValidationError("profile must contain finite nonnegative numbers")
                _float32_bytes(value)
    return used


def _validated_records(records: Iterable[dict[str, Any]], require_gold: bool,
                       *, check_split_isolation: bool = True) -> list[dict[str, Any]]:
    if not isinstance(require_gold, bool):
        raise ValidationError("require_gold must be boolean")
    materialized = list(records)
    if not materialized:
        raise ValidationError("ATR records must contain at least one response")
    response_ids: set[str] = set()
    step_ids: set[str] = set()
    case_splits: dict[str, str] = {}
    contract: dict[str, Any] | None = None
    used = 0
    for record in materialized:
        _fields(record, _RECORD_FIELDS, "ATR response")
        if record["schema_version"] != ATR_SCHEMA_VERSION:
            raise ValidationError(f"schema_version must equal {ATR_SCHEMA_VERSION}")
        response_id = _string(record["response_id"], "response_id")
        case_id = _string(record["case_id"], "case_id")
        _string(record["task_id"], "task_id")
        split = _string(record["split"], "split")
        if split not in ATR_SPLITS:
            raise ValidationError("ATR split must be train, validation, or test")
        if response_id in response_ids:
            raise ValidationError(f"duplicate ATR response_id: {response_id}")
        response_ids.add(response_id)
        if check_split_isolation and case_id in case_splits and case_splits[case_id] != split:
            raise ValidationError(f"ATR case split leakage: {case_id}")
        case_splits[case_id] = split
        _string(record["source_kind"], "source_kind")
        if record["source_kind"] not in {"synthetic_fixture", "reconstructed_recorded"}:
            raise ValidationError("ATR source_kind must be synthetic_fixture or reconstructed_recorded")
        provenance = record["provenance"]
        if not isinstance(provenance, dict):
            raise ValidationError("ATR provenance must be a primitive JSON object")
        _primitive(provenance, "provenance")
        _validate_contract(record)
        if record["source_kind"] == "synthetic_fixture":
            if provenance.get("synthetic") is not True:
                raise ValidationError("synthetic_fixture requires provenance.synthetic=true")
        else:
            for field in ("collection_run_id", "split_assignment_id", "alignment_evidence_id"):
                _string(provenance.get(field), f"provenance.{field}")
            source_digest = provenance.get("profile_source_sha256")
            if not isinstance(source_digest, str) or len(source_digest) != 64 or any(c not in "0123456789abcdef" for c in source_digest.lower()):
                raise ValidationError("provenance.profile_source_sha256 must be a 64-character hexadecimal digest")
            for revision in (record["tokenizer"]["revision"], record["feature_contract"]["reconstruction"]["revision"]):
                if revision.strip().lower() in {"unknown", "unassigned"}:
                    raise ValidationError("reconstructed_recorded requires an explicit metadata revision")
        current = {key: record[key] for key in _CONTRACT_FIELDS}
        if contract is not None and _digest(current) != _digest(contract):
            raise ValidationError("one ATR run requires identical task/tokenizer/alignment/feature contracts")
        contract = current
        steps = record["steps"]
        if not isinstance(steps, list) or not steps:
            raise ValidationError("ATR steps must be a non-empty array for a complete response")
        indices: set[int] = set()
        for step in steps:
            _fields(step, {"step_id", "step_index", "gold_token_id", "profile"}, "ATR step")
            step_id = _string(step["step_id"], "step_id")
            if step_id in step_ids:
                raise ValidationError(f"duplicate ATR step_id: {step_id}")
            step_ids.add(step_id)
            index = _integer(step["step_index"], "step_index")
            if index in indices:
                raise ValidationError("duplicate ATR step_index within response")
            indices.add(index)
            gold = step["gold_token_id"]
            if gold is None:
                if split == "train" or require_gold:
                    raise ValidationError("ATR gold_token_id is required for training/evaluation")
            elif _integer(gold, "gold_token_id") >= record["tokenizer"]["vocab_size"]:
                raise ValidationError("gold_token_id is outside tokenizer.vocab_size")
            if step["profile"] is None:
                if split == "train":
                    raise ValidationError("ATR training steps require an observed profile")
            else:
                used = _validate_profile(step["profile"], record, used)
        if indices != set(range(len(steps))):
            raise ValidationError("ATR step_index must be contiguous and zero-based for a complete response")
    result = deepcopy(materialized)
    result.sort(key=lambda item: item["response_id"])
    for record in result:
        record["steps"].sort(key=lambda item: item["step_index"])
    if check_split_isolation:
        _snapshot(result, check_payload_isolation=True)
    return result


def validate_atr_records(records: Iterable[dict[str, Any]], require_gold: bool = False) -> list[dict[str, Any]]:
    """Return independent canonical copies; missing test observations retain steps.

    Training always requires gold IDs and observed profiles. ``require_gold``
    additionally requires gold for every validation/test step before evaluation.
    """
    return _validated_records(records, require_gold)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _parse_json(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValidationError(f"invalid JSON number: {value}")
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant)


def load_atr_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON array, one response object, or JSONL within fixed input budgets."""
    source = Path(path)
    if source.stat().st_size > MAX_ATR_FILE_BYTES:
        raise ValidationError("ATR file byte budget exceeded")
    with source.open("rb") as handle:
        raw = handle.read(MAX_ATR_FILE_BYTES + 1)
    if len(raw) > MAX_ATR_FILE_BYTES:
        raise ValidationError("ATR file byte budget exceeded")
    try:
        text = raw.decode("utf-8")
        if not text.strip():
            raise ValidationError("ATR records must contain at least one response")
        try:
            value = _parse_json(text)
        except json.JSONDecodeError:
            if text.lstrip().startswith("["):
                raise
            value = [_parse_json(line) for line in text.splitlines() if line.strip()]
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            raise ValidationError("ATR manifest must contain response objects")
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f"invalid ATR JSON: {exc}") from exc
    return validate_atr_records(value)


def atr_record_contract(record: dict[str, Any]) -> dict[str, Any]:
    """Extract a validated semantic contract independent of response labels/IDs."""
    canonical = validate_atr_records([record])[0]
    return {key: deepcopy(canonical[key]) for key in sorted(_CONTRACT_FIELDS)}


def freeze_atr_vocabulary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Freeze sorted candidate token IDs using train responses only; allow test OOV."""
    canonical = validate_atr_records(records)
    training = [record for record in canonical if record["split"] == "train"]
    if not training:
        raise ValidationError("cannot freeze ATR vocabulary without train responses")
    token_ids = sorted({step["gold_token_id"] for record in training for step in record["steps"]})
    if len(token_ids) < 2:
        raise ValidationError("ATR training requires at least two train token classes")
    contract = atr_record_contract(training[0])
    vocabulary = {
        "schema_version": ATR_VOCABULARY_VERSION,
        "task_id": contract["task_id"], "tokenizer": contract["tokenizer"],
        "source_split": "train", "frozen": True, "token_ids": token_ids,
    }
    vocabulary["sha256"] = _digest(vocabulary)
    return vocabulary


def validate_atr_vocabulary(vocabulary: dict[str, Any], contract: dict[str, Any] | None = None) -> set[int]:
    _fields(vocabulary, _VOCABULARY_FIELDS, "ATR vocabulary")
    if vocabulary["schema_version"] != ATR_VOCABULARY_VERSION:
        raise ValidationError("invalid ATR vocabulary schema_version")
    if vocabulary["source_split"] != "train" or vocabulary["frozen"] is not True:
        raise ValidationError("ATR vocabulary must be frozen from train split")
    _string(vocabulary["task_id"], "vocabulary.task_id")
    tokenizer = _tokenizer(vocabulary["tokenizer"])
    token_ids = vocabulary["token_ids"]
    if not isinstance(token_ids, list) or len(token_ids) < 2:
        raise ValidationError("ATR vocabulary requires at least two token IDs")
    for token in token_ids:
        if _integer(token, "vocabulary token_id") >= tokenizer["vocab_size"]:
            raise ValidationError("ATR vocabulary token ID is outside tokenizer.vocab_size")
    if token_ids != sorted(set(token_ids)):
        raise ValidationError("ATR vocabulary token IDs must be unique and numerically sorted")
    if vocabulary["sha256"] != _digest({key: value for key, value in vocabulary.items() if key != "sha256"}):
        raise ValidationError("ATR vocabulary digest mismatch")
    if contract is not None:
        _fields(contract, _CONTRACT_FIELDS, "ATR record contract")
        if contract["schema_version"] != ATR_SCHEMA_VERSION:
            raise ValidationError("invalid ATR record contract schema_version")
        _string(contract["task_id"], "contract.task_id")
        _validate_contract(contract)
        if contract["task_id"] != vocabulary["task_id"] or contract["tokenizer"] != tokenizer:
            raise ValidationError("ATR vocabulary task/tokenizer contract mismatch")
    return set(token_ids)


def _snapshot(records: list[dict[str, Any]], *, check_payload_isolation: bool) -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    profile_splits: dict[str, str] = {}
    for record in records:
        profiles = [step["profile"] for step in record["steps"]]
        digest = _profile_digest(profiles)
        if check_payload_isolation and digest is not None:
            previous = profile_splits.get(digest)
            if previous is not None and previous != record["split"]:
                raise ValidationError("ATR complete response profile split leakage")
            profile_splits[digest] = record["split"]
        snapshot.append({
            "response_id": record["response_id"], "case_id": record["case_id"],
            "split": record["split"], "task_id": record["task_id"],
            "step_ids": [step["step_id"] for step in record["steps"]],
            "gold_token_ids": [step["gold_token_id"] for step in record["steps"]],
            "profile_sha256": digest,
        })
    return snapshot


def atr_record_snapshot(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Capture IDs/gold and semantic float32 profile digest without profiles."""
    return _snapshot(validate_atr_records(records), check_payload_isolation=True)


def assign_atr_grouped_splits(
    records: Iterable[dict[str, Any]], *, train_ratio: float = 0.8,
    validation_ratio: float = 0.1, test_ratio: float = 0.1, seed: str = "janus-atr-v1",
) -> list[dict[str, Any]]:
    """Replace assignments with deterministic case groups (and disjoint responses).

    The 8:1:1 default follows the paper's response disjointness. Keeping all
    responses for one case together is an explicit conservative reconstruction.
    Generated assignments receive a digest ID and configuration evidence; the
    supplied prior ID is preserved as parent_split_assignment_id. These IDs
    record this helper's operation and do not authenticate external metadata.
    """
    _string(seed, "split seed")
    for name, value in (("train_ratio", train_ratio), ("validation_ratio", validation_ratio), ("test_ratio", test_ratio)):
        try:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        except (OverflowError, TypeError):
            valid = False
        if not valid:
            raise ValidationError(f"{name} must be finite and nonnegative")
    canonical = _validated_records(records, False, check_split_isolation=False)
    assigned = assign_grouped_splits(canonical, group_field="case_id", train_ratio=train_ratio,
                                     validation_ratio=validation_ratio, test_ratio=test_ratio, seed=seed)
    assignments_digest = _digest([
        {"response_id": record["response_id"], "case_id": record["case_id"], "split": record["split"]}
        for record in sorted(assigned, key=lambda item: item["response_id"])
    ])
    split_contract = {
        "schema_version": "janus.atr.split.v1", "group_field": "case_id", "seed": seed,
        "ratios": {"train": float(train_ratio), "validation": float(validation_ratio), "test": float(test_ratio)},
        "assignments_sha256": assignments_digest,
    }
    assignment_id = "atr-grouped-" + _digest(split_contract)
    for record in assigned:
        provenance = record["provenance"]
        previous_id = provenance.get("split_assignment_id")
        if previous_id is not None:
            provenance["parent_split_assignment_id"] = previous_id
        provenance["split_assignment_id"] = assignment_id
        provenance["generated_split_contract"] = deepcopy(split_contract)
    return validate_atr_records(assigned)
