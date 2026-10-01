"""Strict offline probe, mapping, calibration and oracle-reference contracts.

Validation checks declared evidence and consistency. It does not authenticate
physical collection, recover victim addresses, or infer phase/gold from probes.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from .schema import ValidationError


SCHEMA_VERSION = "janus.probe.run.v1"
WORKLOAD_SCHEMA_VERSION = "janus.controlled_workload.v1"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_OBSERVATIONS = 1_000_000
MAX_RECONSTRUCTED_CELLS = 1_000_000
MAX_PRIMITIVE_NODES = 8_000_000
MAX_JSON_DEPTH = 32
MAX_STRING_BYTES = 16_384
MAX_ID_CHARS = 256
MIN_INT = -(2**63)
MAX_INT = 2**63 - 1

RUN_FIELDS = {
    "schema_version", "run_id", "source_kind", "provenance", "task_id",
    "case_id", "sample_id", "response_id", "split", "split_assignment_id",
    "allocation_epoch", "clock", "layout", "calibration", "probe_rounds",
    "phase_reference", "gold",
}
WORKLOAD_FIELDS = {"schema_version", "config", "split_contract", "tokenizer", "runs"}


def _primitive_guard(value: Any) -> None:
    """Check bounded builtin JSON types and finite numbers before copying."""
    nodes = 0
    byte_count = 0
    ancestors: set[int] = set()

    def walk(item: Any, depth: int) -> None:
        nonlocal nodes, byte_count
        nodes += 1
        if nodes > MAX_PRIMITIVE_NODES or depth > MAX_JSON_DEPTH:
            raise ValidationError("probe JSON exceeds the node/depth budget")
        kind = type(item)
        if item is None:
            byte_count += 4
        elif kind is bool:
            byte_count += 5
        elif kind is int:
            if not MIN_INT <= item <= MAX_INT:
                raise ValidationError("probe JSON integer exceeds the signed 64-bit budget")
            byte_count += len(str(item))
        elif kind is float:
            if not math.isfinite(item):
                raise ValidationError("probe JSON numbers must be finite")
            byte_count += len(repr(item))
        elif kind is str:
            try:
                encoded_size = len(item.encode("utf-8"))
            except UnicodeError as exc:
                raise ValidationError("probe JSON strings must be valid Unicode") from exc
            if encoded_size > MAX_STRING_BYTES:
                raise ValidationError("probe JSON string exceeds the string budget")
            # Conservative bound on compact UTF-8 JSON escaping.
            byte_count += encoded_size + 2
            byte_count += sum(5 if ord(char) < 32 else 1 if char in '"\\' else 0 for char in item)
        elif kind in (dict, list):
            identity = id(item)
            if identity in ancestors:
                raise ValidationError("probe JSON must not contain cycles")
            ancestors.add(identity)
            byte_count += 2 + max(0, len(item) - 1)
            try:
                if kind is dict:
                    byte_count += len(item)  # colons
                    for key, child in item.items():
                        if type(key) is not str:
                            raise ValidationError("probe JSON object keys must be strings")
                        walk(key, depth + 1)
                        walk(child, depth + 1)
                else:
                    for child in item:
                        walk(child, depth + 1)
            finally:
                ancestors.remove(identity)
        else:
            raise ValidationError("probe JSON requires builtin primitive JSON types")
        if byte_count > MAX_FILE_BYTES:
            raise ValidationError("probe JSON exceeds the byte budget")

    walk(value, 0)


def _fields(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValidationError(f"{path} fields must equal {sorted(expected)}")
    return value


def _string(value: Any, path: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > MAX_ID_CHARS:
        raise ValidationError(f"{path} must be a bounded non-empty string")
    return value


def _integer(value: Any, path: str, minimum: int = 0, maximum: int = MAX_INT) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError(f"{path} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite(value: Any, path: str, *, nonnegative: bool = False) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValidationError(f"{path} must be a finite number")
    if nonnegative and value < 0:
        raise ValidationError(f"{path} must be nonnegative")
    return value


def _list(value: Any, path: str, *, maximum: int = MAX_OBSERVATIONS) -> list[Any]:
    if type(value) is not list or not 0 < len(value) <= maximum:
        raise ValidationError(f"{path} must be a non-empty bounded array")
    return value


def _unique_strings(value: Any, path: str) -> list[str]:
    items = _list(value, path)
    for item in items:
        _string(item, path)
    if len(items) != len(set(items)):
        raise ValidationError(f"{path} contains duplicate IDs")
    return items


def _validate_run(run: Any) -> tuple[int, int]:
    _fields(run, RUN_FIELDS, "probe run")
    if run["schema_version"] != SCHEMA_VERSION:
        raise ValidationError(f"probe run schema_version must be {SCHEMA_VERSION}")
    for field in (
        "run_id", "task_id", "case_id", "sample_id", "response_id",
        "split_assignment_id", "allocation_epoch",
    ):
        _string(run[field], field)
    if run["split"] not in ("train", "validation", "test"):
        raise ValidationError("probe run split must be train, validation or test")
    synthetic = run["source_kind"] == "synthetic_probe_simulation"
    physical = run["source_kind"] == "physical_probe_recording"
    if not synthetic and not physical:
        raise ValidationError("raw source_kind must declare synthetic or physical probes")
    provenance = run["provenance"]
    if type(provenance) is not dict:
        raise ValidationError("provenance must be a primitive object")
    if synthetic and provenance.get("synthetic") is not True:
        raise ValidationError("synthetic probes require provenance.synthetic=true")
    if physical:
        if provenance.get("synthetic") is True:
            raise ValidationError("physical probes cannot claim synthetic provenance")
        for field in ("collection_run_id", "collector_revision"):
            _string(provenance.get(field), f"provenance.{field}")

    clock = _fields(
        run["clock"],
        {"unit", "probe_collection_start_ns", "reference_collection_start_ns"},
        "clock",
    )
    if clock["unit"] != "ns":
        raise ValidationError("clock.unit must be ns")
    _finite(clock["probe_collection_start_ns"], "clock.probe_collection_start_ns")
    _finite(clock["reference_collection_start_ns"], "clock.reference_collection_start_ns")

    layout = _fields(
        run["layout"],
        {"source_kind", "calibration_id", "allocation_epoch", "layer_ids",
         "kv_head_ids", "token_width", "page_entries"},
        "layout",
    )
    calibration = _fields(
        run["calibration"],
        {"source_kind", "calibration_id", "allocation_epoch", "probes"},
        "calibration",
    )
    layout_source = "synthetic_layout" if synthetic else "calibrated_logical_mapping"
    calibration_source = "synthetic_calibration" if synthetic else "physical_contention_calibration"
    if layout["source_kind"] != layout_source or calibration["source_kind"] != calibration_source:
        raise ValidationError("probe/layout/calibration source classes are incompatible")
    for section_name, section in (("layout", layout), ("calibration", calibration)):
        _string(section["calibration_id"], f"{section_name}.calibration_id")
        _string(section["allocation_epoch"], f"{section_name}.allocation_epoch")
        if section["allocation_epoch"] != run["allocation_epoch"]:
            raise ValidationError("run/layout/calibration allocation_epoch mismatch")
    if layout["calibration_id"] != calibration["calibration_id"]:
        raise ValidationError("layout/calibration calibration_id mismatch")

    layers = _unique_strings(layout["layer_ids"], "layout.layer_ids")
    heads = _unique_strings(layout["kv_head_ids"], "layout.kv_head_ids")
    token_width = _integer(
        layout["token_width"], "layout.token_width", 1, MAX_RECONSTRUCTED_CELLS
    )
    profile_cells = len(layers) * len(heads) * token_width
    if profile_cells > MAX_RECONSTRUCTED_CELLS:
        raise ValidationError("layout exceeds the reconstructed tensor budget")
    pages = _list(layout["page_entries"], "layout.page_entries")
    page_ids: set[str] = set()
    page_probes: set[str] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for page in pages:
        _fields(
            page,
            {"page_id", "probe_id", "layer_id", "kv_head_id", "page_order",
             "token_start", "token_count"},
            "layout.page_entries entry",
        )
        for field in ("page_id", "probe_id", "layer_id", "kv_head_id"):
            _string(page[field], f"page.{field}")
        if page["page_id"] in page_ids or page["probe_id"] in page_probes:
            raise ValidationError("layout contains duplicate page_id or probe_id")
        page_ids.add(page["page_id"])
        page_probes.add(page["probe_id"])
        if page["layer_id"] not in layers or page["kv_head_id"] not in heads:
            raise ValidationError("page layer/head is absent from the declared layout axes")
        _integer(page["page_order"], "page.page_order", 0, token_width - 1)
        _integer(page["token_start"], "page.token_start", 0, token_width - 1)
        _integer(page["token_count"], "page.token_count", 1, token_width)
        grouped.setdefault((page["layer_id"], page["kv_head_id"]), []).append(page)
    if len(grouped) != len(layers) * len(heads):
        raise ValidationError("layout must cover every declared layer/head pair")
    for entries in grouped.values():
        ordered = sorted(entries, key=lambda page: page["page_order"])
        position = 0
        for order, page in enumerate(ordered):
            if page["page_order"] != order or page["token_start"] != position:
                raise ValidationError("page order and token intervals must be contiguous from zero")
            position += page["token_count"]
        if position != token_width:
            raise ValidationError("page token intervals must exactly cover token_width")

    calibration_probes: dict[str, dict[str, Any]] = {}
    for probe in _list(calibration["probes"], "calibration.probes"):
        _fields(
            probe,
            {"probe_id", "translation_threshold_ns", "probe_set_line_count"},
            "calibration probe",
        )
        probe_id = _string(probe["probe_id"], "calibration.probe_id")
        if probe_id in calibration_probes:
            raise ValidationError("calibration contains duplicate probe_id")
        _finite(probe["translation_threshold_ns"], "translation_threshold_ns", nonnegative=True)
        _integer(probe["probe_set_line_count"], "probe_set_line_count", 1)
        calibration_probes[probe_id] = probe
    if set(calibration_probes) != page_probes:
        raise ValidationError("layout/calibration probes must form a bijection")

    observation_count = 0
    round_ids: set[str] = set()
    last_timestamp = None
    for round_record in _list(run["probe_rounds"], "probe_rounds"):
        _fields(round_record, {"round_id", "timestamp_ns", "observations"}, "raw probe round")
        round_id = _string(round_record["round_id"], "round_id")
        if round_id in round_ids:
            raise ValidationError("probe_rounds contains duplicate round_id")
        round_ids.add(round_id)
        timestamp = _finite(round_record["timestamp_ns"], "round.timestamp_ns")
        if last_timestamp is not None and timestamp <= last_timestamp:
            raise ValidationError("probe round timestamps must be strictly ascending")
        last_timestamp = timestamp
        observations = _list(round_record["observations"], "round.observations")
        observation_count += len(observations)
        if observation_count > MAX_OBSERVATIONS:
            raise ValidationError("raw probes exceed the observation budget")
        observed: set[str] = set()
        for observation in observations:
            _fields(
                observation,
                {"probe_id", "reload_latency_ns", "evicted_probe_lines"},
                "raw probe observation",
            )
            probe_id = _string(observation["probe_id"], "observation.probe_id")
            if probe_id not in calibration_probes or probe_id in observed:
                raise ValidationError("raw observations contain an unknown or duplicate probe_id")
            observed.add(probe_id)
            _finite(observation["reload_latency_ns"], "reload_latency_ns", nonnegative=True)
            _integer(
                observation["evicted_probe_lines"], "evicted_probe_lines", 0,
                calibration_probes[probe_id]["probe_set_line_count"],
            )
        if observed != page_probes:
            raise ValidationError("every raw round must contain all calibrated probes")

    reference = _fields(
        run["phase_reference"],
        {"source_kind", "reference_method", "alignment_evidence_id", "boundaries",
         "collection_end_ns"},
        "phase_reference",
    )
    if reference["source_kind"] != "oracle_annotation":
        raise ValidationError("phase_reference must declare oracle_annotation")
    _string(reference["reference_method"], "phase_reference.reference_method")
    _string(reference["alignment_evidence_id"], "phase_reference.alignment_evidence_id")
    boundaries = _list(reference["boundaries"], "phase_reference.boundaries")
    last_timestamp = None
    reference_steps = []
    reference_step_ids: set[str] = set()
    for index, boundary in enumerate(boundaries):
        _fields(
            boundary, {"timestamp_ns", "phase", "step_id", "step_index"},
            "phase reference boundary",
        )
        timestamp = _finite(boundary["timestamp_ns"], "boundary.timestamp_ns")
        if last_timestamp is not None and timestamp <= last_timestamp:
            raise ValidationError("phase reference timestamps must be strictly ascending")
        last_timestamp = timestamp
        if index == 0:
            if boundary["phase"] != "prefill" or boundary["step_id"] is not None or boundary["step_index"] is not None:
                raise ValidationError("first reference boundary must be prefill with null step fields")
        else:
            if boundary["phase"] != "decoding":
                raise ValidationError("later reference boundaries must be decoding")
            step_id = _string(boundary["step_id"], "boundary.step_id")
            _integer(boundary["step_index"], "boundary.step_index")
            if boundary["step_index"] != index - 1:
                raise ValidationError("reference decoding step indices must be contiguous from zero")
            if step_id in reference_step_ids:
                raise ValidationError("phase reference contains duplicate step_id")
            reference_step_ids.add(step_id)
            reference_steps.append((step_id, index - 1))
    collection_end = _finite(reference["collection_end_ns"], "phase_reference.collection_end_ns")
    if collection_end <= last_timestamp:
        raise ValidationError("phase_reference collection_end_ns must follow the last boundary")

    gold = _fields(
        run["gold"],
        {"source_kind", "oracle_method", "attributes", "tokenizer", "alignment", "steps"},
        "gold",
    )
    if gold["source_kind"] != "oracle_annotation":
        raise ValidationError("gold must declare oracle_annotation")
    _string(gold["oracle_method"], "gold.oracle_method")
    _fields(gold["attributes"], {"toy_topic"}, "gold.attributes")
    if gold["attributes"]["toy_topic"] not in ("alpha", "beta"):
        raise ValidationError("gold.attributes.toy_topic must be alpha or beta")
    tokenizer = _fields(gold["tokenizer"], {"name", "revision", "vocab_size"}, "gold.tokenizer")
    _string(tokenizer["name"], "tokenizer.name")
    _string(tokenizer["revision"], "tokenizer.revision")
    vocab_size = _integer(tokenizer["vocab_size"], "tokenizer.vocab_size", 1)
    alignment = _fields(
        gold["alignment"],
        {"step_index_base", "profile_predicts", "bos_included", "eos_included",
         "special_tokens", "response_scope"},
        "gold.alignment",
    )
    if (
        type(alignment["step_index_base"]) is not int or alignment["step_index_base"] != 0
        or alignment["profile_predicts"] != "same_index_output_token"
        or alignment["bos_included"] is not False or alignment["eos_included"] is not False
        or alignment["special_tokens"] != "excluded"
        or alignment["response_scope"] != "complete_response"
    ):
        raise ValidationError("gold alignment must explicitly declare complete same-index output tokens")
    gold_steps = _list(gold["steps"], "gold.steps")
    if len(gold_steps) != len(reference_steps):
        raise ValidationError("gold/reference must declare exactly the same decoding steps")
    for index, step in enumerate(gold_steps):
        _fields(step, {"step_id", "step_index", "gold_token_id"}, "gold step")
        _string(step["step_id"], "gold.step_id")
        _integer(step["step_index"], "gold.step_index")
        _integer(step["gold_token_id"], "gold.gold_token_id", 0, vocab_size - 1)
        if (step["step_id"], step["step_index"]) != reference_steps[index]:
            raise ValidationError("gold/reference decoding step identity or order mismatch")
    reconstructed_cells = profile_cells * (len(gold_steps) + 1)
    if reconstructed_cells > MAX_RECONSTRUCTED_CELLS:
        raise ValidationError("response exceeds the reconstructed tensor budget")
    return observation_count, reconstructed_cells


def validate_probe_run(run: Any) -> dict[str, Any]:
    """Validate one raw run and return an independent builtin JSON copy."""
    _primitive_guard(run)
    _validate_run(run)
    return deepcopy(run)


def validate_probe_runs(runs: Any) -> list[dict[str, Any]]:
    """Validate bounded runs, global IDs and case split isolation before copy."""
    _primitive_guard(runs)
    _list(runs, "probe runs")
    identities: dict[str, set[str]] = {
        "run_id": set(), "sample_id": set(), "response_id": set(), "step_id": set()
    }
    case_splits: dict[str, str] = {}
    observations = 0
    cells = 0
    for run in runs:
        count, run_cells = _validate_run(run)
        observations += count
        cells += run_cells
        if observations > MAX_OBSERVATIONS:
            raise ValidationError("probe runs exceed the total observation budget")
        if cells > MAX_RECONSTRUCTED_CELLS:
            raise ValidationError("probe runs exceed the total reconstructed tensor budget")
        for field in ("run_id", "sample_id", "response_id"):
            identifier = run[field]
            if identifier in identities[field]:
                raise ValidationError(f"probe runs contain duplicate {field}")
            identities[field].add(identifier)
        for step in run["gold"]["steps"]:
            if step["step_id"] in identities["step_id"]:
                raise ValidationError("probe runs contain duplicate step_id")
            identities["step_id"].add(step["step_id"])
        case_id = run["case_id"]
        if case_id in case_splits and case_splits[case_id] != run["split"]:
            raise ValidationError("probe runs contain case split leakage")
        case_splits[case_id] = run["split"]
    return deepcopy(runs)


def canonical_sha256(value: Any) -> str:
    """Hash canonical finite UTF-8 JSON; integrity is not authentication."""
    _primitive_guard(value)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_FILE_BYTES:
        raise ValidationError("canonical probe JSON exceeds the byte budget")
    return hashlib.sha256(encoded).hexdigest()


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("probe JSON contains duplicate object fields")
        result[key] = value
    return result


def _json_integer(text: str) -> int:
    if len(text.lstrip("-")) > 19:
        raise ValidationError("probe JSON integer exceeds the signed 64-bit budget")
    value = int(text)
    if not MIN_INT <= value <= MAX_INT:
        raise ValidationError("probe JSON integer exceeds the signed 64-bit budget")
    return value


def _json_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValidationError("probe JSON numbers must be finite")
    return value


def _json_constant(text: str) -> None:
    raise ValidationError(f"probe JSON forbids non-finite constant {text}")


def _parse_json(text: str) -> Any:
    try:
        return json.loads(
            text, object_pairs_hook=_json_object, parse_int=_json_integer,
            parse_float=_json_float, parse_constant=_json_constant,
        )
    except RecursionError as exc:
        raise ValidationError("probe JSON exceeds the parser depth budget") from exc


def load_probe_runs(path: str | Path) -> list[dict[str, Any]]:
    """Read raw runs or validate and extract a controlled workload bundle.

    Recognized workload bundles also require their owner's config, tokenizer
    and frozen split-assignment proof validation before exposing the runs.
    """
    input_path = Path(path)
    try:
        if not input_path.is_file() or not 0 < input_path.stat().st_size <= MAX_FILE_BYTES:
            raise ValidationError("probe input exceeds the file-size budget or is absent")
        with input_path.open("rb") as handle:
            data = handle.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValidationError("probe input exceeds the file-size budget")
        text = data.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError("probe input must be a readable UTF-8 file") from exc
    try:
        parsed = _parse_json(text)
    except json.JSONDecodeError:
        parsed = []
        try:
            for line in text.splitlines():
                if line.strip():
                    parsed.append(_parse_json(line))
                    if len(parsed) > MAX_OBSERVATIONS:
                        raise ValidationError("probe JSONL exceeds the run budget")
        except json.JSONDecodeError as exc:
            raise ValidationError("probe input is neither valid JSON nor JSONL") from exc
    _primitive_guard(parsed)
    if type(parsed) is dict:
        if parsed.get("schema_version") == WORKLOAD_SCHEMA_VERSION:
            _fields(parsed, WORKLOAD_FIELDS, "controlled workload wrapper")
            # Lazy import avoids the workload -> probe validator dependency
            # cycle while preserving the wrapper's frozen split proof.
            from .workload import validate_controlled_workload

            return validate_controlled_workload(parsed)["runs"]
        parsed = [parsed]
    return validate_probe_runs(parsed)
