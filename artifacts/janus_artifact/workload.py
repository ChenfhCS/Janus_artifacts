"""Reproducible CPU toy KV execution and explicitly synthetic probe observables.

The timing, eviction budget, query rules and noise are named reconstruction
choices. No physical probe, model, GPU address, or paper measurement is produced.
Oracle phase/gold objects are separate from the raw ``probe_rounds`` entries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .probe_contract import canonical_sha256, validate_probe_runs
from .schema import ValidationError

WORKLOAD_SCHEMA_VERSION = "janus.controlled_workload.v1"
SPLIT_SCHEMA_VERSION = "janus.controlled_split.v1"
MAX_WORKLOAD_FILE_BYTES = 64 * 1024 * 1024
MAX_WORKLOAD_OBSERVATIONS = 1_000_000
SIMULATION_REVISION = "janus.cpu-toy-probe.v1"
SYNTHETIC_TRANSLATION_THRESHOLD_NS = 150.0
SYNTHETIC_HIT_LATENCY_NS = 100.0
SYNTHETIC_MISS_LATENCY_NS = 200.0
SYNTHETIC_PROBE_SET_LINE_COUNT = 8
SYNTHETIC_ROUND_INTERVAL_NS = 100
SYNTHETIC_PROBE_START_NS = 1000
SYNTHETIC_REFERENCE_START_NS = 2000
_BUNDLE_FIELDS = {"schema_version", "config", "split_contract", "tokenizer", "runs"}
_TOKENIZER = {
    "name": "janus-cpu-toy-symbolic", "revision": "toy-sign-integer-v1", "vocab_size": 24,
    "ordered_symbolic_table": [
        {"token_id": 11, "symbol": "toy_alpha"},
        {"token_id": 17, "symbol": "toy_beta"},
        {"token_id": 23, "symbol": "toy_zero_sum"},
    ],
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate workload JSON key: {key}")
        result[key] = value
    return result


def _load_bounded_json(path: str | Path) -> Any:
    source = Path(path)
    if source.stat().st_size > MAX_WORKLOAD_FILE_BYTES:
        raise ValidationError("controlled workload file byte budget exceeded")
    with source.open("rb") as handle:
        raw = handle.read(MAX_WORKLOAD_FILE_BYTES + 1)
    if len(raw) > MAX_WORKLOAD_FILE_BYTES:
        raise ValidationError("controlled workload file byte budget exceeded")
    def reject_constant(value: str) -> None:
        raise ValidationError(f"invalid nonfinite workload JSON number: {value}")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f"invalid controlled workload JSON: {exc}") from exc


@dataclass(frozen=True)
class WorkloadConfig:
    """Explicit toy parameters; defaults are synthetic choices, not constants from the paper."""
    case_count: int = 24
    steps_per_response: int = 3
    seed: int = 19
    split_seed: str = "janus-upstream-v1"
    layers: int = 4
    kv_heads: int = 4
    pages_per_head: int = 4
    tokens_per_page: int = 2
    prefill_rounds: int = 4
    decode_rounds: int = 5
    noise_rate: float = 0.02

    def validate(self) -> None:
        for name in ("case_count", "steps_per_response", "layers", "kv_heads", "pages_per_head",
                     "tokens_per_page", "prefill_rounds", "decode_rounds"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValidationError(f"workload {name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not -(1 << 63) <= self.seed < (1 << 63):
            raise ValidationError("workload seed must be a signed int64 integer")
        if not isinstance(self.split_seed, str) or not self.split_seed.strip():
            raise ValidationError("workload split_seed must be nonempty text")
        try:
            if len(self.split_seed.encode("utf-8")) > 16384:
                raise ValidationError("workload split_seed exceeds the string budget")
        except UnicodeError as exc:
            raise ValidationError("workload split_seed must be valid UTF-8") from exc
        try:
            valid_noise = isinstance(self.noise_rate, (int, float)) and not isinstance(self.noise_rate, bool) and math.isfinite(self.noise_rate) and 0 <= self.noise_rate <= 1
        except (OverflowError, TypeError):
            valid_noise = False
        if not valid_noise:
            raise ValidationError("workload noise_rate must be finite and in [0,1]")
        if self.pages_per_head * self.tokens_per_page < 2:
            raise ValidationError("toy signed KV table requires at least two key tokens")
        if self.case_count * self.layers * self.kv_heads * self.pages_per_head * self.tokens_per_page * (self.steps_per_response + 1) > 1_000_000:
            raise ValidationError("workload reconstructed feature budget exceeded")
        observations = self.case_count * self.layers * self.kv_heads * self.pages_per_head * (
            self.prefill_rounds + self.steps_per_response * self.decode_rounds)
        if observations > MAX_WORKLOAD_OBSERVATIONS:
            raise ValidationError("workload observation budget exceeded")
        # Conservative serialized-JSON upper bound for this generator's fixed
        # bounded ID/string/number formats, checked before allocating observations.
        # Budget coefficients are resource limits, not physical simulator values.
        probes = self.layers * self.kv_heads * self.pages_per_head
        rounds = self.prefill_rounds + self.steps_per_response * self.decode_rounds
        estimated_bytes = (observations * 256 + self.case_count * rounds * 256
                           + self.case_count * probes * 1024
                           + self.case_count * self.steps_per_response * 512
                           + self.case_count * 8192 + 262144)
        if estimated_bytes > MAX_WORKLOAD_FILE_BYTES:
            raise ValidationError("workload conservative serialized byte budget exceeded")

    @classmethod
    def from_dict(cls, value: Any) -> "WorkloadConfig":
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValidationError(f"workload config must explicitly contain all fields: {sorted(cls.__dataclass_fields__)}")
        config = cls(**value)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "WorkloadConfig":
        return cls.from_dict(_load_bounded_json(path))


def toy_tokenizer_contract() -> dict[str, Any]:
    """Return the fixed pre-test symbolic table; no token table is learned from test."""
    return deepcopy(_TOKENIZER)


def toy_token_id_from_aggregate(value: int) -> int:
    """CPU oracle interface for the actual signed integer KV aggregate."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError("toy aggregate must be an integer")
    return 11 if value < 0 else 17 if value > 0 else 23


def integer_kv_aggregate(accesses: Iterable[int], kv_values: list[int]) -> int:
    """Execute integer gathers/additions without a model or floating point oracle."""
    if not isinstance(kv_values, list) or not kv_values or any(not isinstance(value, int) or isinstance(value, bool) for value in kv_values):
        raise ValidationError("toy KV values must be a nonempty integer list")
    try:
        iterator = iter(accesses)
    except TypeError as exc:
        raise ValidationError("toy KV accesses must be an iterable of integer indices") from exc
    total = 0
    count = 0
    for index in iterator:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(kv_values):
            raise ValidationError("toy KV access index is out of range")
        total += kv_values[index]
        count += 1
    if not count:
        raise ValidationError("toy KV aggregation requires at least one access")
    return total


def _hash_integer(*parts: Any) -> int:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest(), "big")


def _case_id(index: int) -> str:
    return f"toy-case-{index:06d}"


def _ids(index: int, config: WorkloadConfig) -> dict[str, Any]:
    return {
        "case_id": _case_id(index), "run_id": f"toy-run-{index:06d}",
        "sample_id": f"toy-prefill-{index:06d}", "response_id": f"toy-response-{index:06d}",
        "step_ids": [f"toy-response-{index:06d}:step-{step:04d}" for step in range(config.steps_per_response)],
    }


def _split_contract(config: WorkloadConfig) -> dict[str, Any]:
    """Freeze case groups before any KV query, raw observation, or gold execution."""
    assignments = []
    for index in range(config.case_count):
        case = _case_id(index)
        bucket = int.from_bytes(hashlib.sha256(f"{config.split_seed}:{case}".encode("utf-8")).digest(), "big") % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        assignments.append({"case_id": case, "split": split})
    contract = {
        "schema_version": SPLIT_SCHEMA_VERSION, "group_field": "case_id",
        "method": "sha256_integer_mod100", "seed": config.split_seed, "bucket_count": 100,
        "thresholds": {"train": 80, "validation": 90, "test": 100},
        "assignments": assignments, "assignments_sha256": canonical_sha256(assignments),
    }
    contract["split_assignment_id"] = "toy-split-" + canonical_sha256(contract)
    return contract


def _layout(index: int, config: WorkloadConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    case = _case_id(index)
    geometry = canonical_sha256({"layers": config.layers, "kv_heads": config.kv_heads,
                                 "pages_per_head": config.pages_per_head, "tokens_per_page": config.tokens_per_page})[:16]
    epoch = f"synthetic-allocation:{case}:{geometry}"
    calibration_id = f"synthetic-calibration:{case}:{geometry}"
    layer_ids = [f"toy-layer-{layer:03d}" for layer in range(config.layers)]
    head_ids = [f"toy-kv-head-{head:03d}" for head in range(config.kv_heads)]
    entries, probes = [], []
    for layer_id in layer_ids:
        for head_id in head_ids:
            for page in range(config.pages_per_head):
                suffix = f"{case}:{layer_id}:{head_id}:page-{page:03d}"
                probe_id = f"synthetic-probe:{suffix}"
                entries.append({"page_id": f"synthetic-page:{suffix}", "probe_id": probe_id,
                                "layer_id": layer_id, "kv_head_id": head_id, "page_order": page,
                                "token_start": page * config.tokens_per_page, "token_count": config.tokens_per_page})
                probes.append({"probe_id": probe_id, "translation_threshold_ns": SYNTHETIC_TRANSLATION_THRESHOLD_NS,
                               "probe_set_line_count": SYNTHETIC_PROBE_SET_LINE_COUNT})
    layout = {"source_kind": "synthetic_layout", "calibration_id": calibration_id, "allocation_epoch": epoch,
              "layer_ids": layer_ids, "kv_head_ids": head_ids,
              "token_width": config.pages_per_head * config.tokens_per_page, "page_entries": entries}
    calibration = {"source_kind": "synthetic_calibration", "calibration_id": calibration_id,
                   "allocation_epoch": epoch, "probes": probes}
    return layout, calibration


def _simulation_provenance(config: WorkloadConfig) -> dict[str, Any]:
    return {
        "synthetic": True, "scientific_result": False, "simulation_revision": SIMULATION_REVISION,
        "simulator": "cpu_toy_latency_eviction_from_integer_kv_accesses",
        "simulation_parameters": {
            "seed": config.seed, "noise_rate": config.noise_rate,
            "noise_rule": "independent_hash_page_event_flip",
            "translation_threshold_ns": SYNTHETIC_TRANSLATION_THRESHOLD_NS,
            "hit_latency_ns": SYNTHETIC_HIT_LATENCY_NS, "miss_latency_ns": SYNTHETIC_MISS_LATENCY_NS,
            "probe_set_line_count": SYNTHETIC_PROBE_SET_LINE_COUNT,
            "round_interval_ns": SYNTHETIC_ROUND_INTERVAL_NS,
            "parameter_status": "named_synthetic_choices_not_paper_physical_constants",
        },
    }


def _query_accesses(config: WorkloadConfig, case_index: int, phase: str, query_index: int,
                    layer_index: int, head_index: int, split: str) -> list[int]:
    width = config.pages_per_head * config.tokens_per_page
    half = width // 2
    key = (config.seed, _case_id(case_index), phase, query_index, layer_index, head_index)
    if phase == "decoding" and split == "test" and query_index == config.steps_per_response - 1:
        # Controlled declared OOV23: gather one -1 and one +1, so the actual sum is zero.
        return [_hash_integer(*key, "negative") % half,
                half + _hash_integer(*key, "positive") % (width - half)]
    lower = case_index % 2 == 0
    start, count = (0, half) if lower else (half, width - half)
    access_count = 1 + _hash_integer(*key, "count") % (SYNTHETIC_PROBE_SET_LINE_COUNT if phase == "prefill" else 2)
    return [start + _hash_integer(*key, access_index) % count for access_index in range(access_count)]


def _simulate_observation(config: WorkloadConfig, ids: dict[str, Any], phase: str, query_index: int,
                          vote_index: int, entry: dict[str, Any], access_count: int) -> dict[str, Any]:
    active = access_count > 0
    noise_value = _hash_integer(config.seed, ids["case_id"], phase, query_index, vote_index, entry["probe_id"], "noise") / (1 << 256)
    flipped = config.noise_rate == 1.0 or noise_value < config.noise_rate
    if flipped:
        active = not active
    footprint = min(SYNTHETIC_PROBE_SET_LINE_COUNT, access_count) if active and not flipped else 0
    if active and flipped:
        footprint = 1 + _hash_integer(config.seed, ids["case_id"], entry["probe_id"], query_index, vote_index, "noise_footprint") % SYNTHETIC_PROBE_SET_LINE_COUNT
    return {"probe_id": entry["probe_id"],
            "reload_latency_ns": SYNTHETIC_MISS_LATENCY_NS if active else SYNTHETIC_HIT_LATENCY_NS,
            "evicted_probe_lines": footprint}


def _expected_observations(config: WorkloadConfig, case_index: int, split: str,
                           ids: dict[str, Any], layout: dict[str, Any],
                           phase: str, query_index: int, vote_index: int) -> list[dict[str, Any]]:
    """Derive raw simulator observables solely from frozen workload/config, never gold."""
    head_positions = {(layer, head): (layer_index, head_index)
                      for layer_index, layer in enumerate(layout["layer_ids"])
                      for head_index, head in enumerate(layout["kv_head_ids"])}
    accesses = {position: _query_accesses(config, case_index, phase, query_index, *position, split)
                for position in head_positions.values()}
    observations = []
    for entry in layout["page_entries"]:
        selected = accesses[head_positions[(entry["layer_id"], entry["kv_head_id"])]]
        count = sum(entry["token_start"] <= token < entry["token_start"] + entry["token_count"] for token in selected)
        observations.append(_simulate_observation(config, ids, phase, query_index, vote_index, entry, count))
    return observations


def _generate_run(index: int, split: str, assignment_id: str, config: WorkloadConfig) -> dict[str, Any]:
    ids = _ids(index, config)
    layout, calibration = _layout(index, config)
    width = layout["token_width"]
    kv_values = [-1 if token < width // 2 else 1 for token in range(width)]
    head_positions = {(layer, head): (layer_index, head_index)
                      for layer_index, layer in enumerate(layout["layer_ids"])
                      for head_index, head in enumerate(layout["kv_head_ids"])}
    rounds = []
    gold_steps = []
    def append_round(phase: str, query_index: int, vote_index: int) -> None:
        round_index = len(rounds)
        observations = _expected_observations(config, index, split, ids, layout, phase, query_index, vote_index)
        rounds.append({"round_id": f"{ids['run_id']}:probe-round-{round_index:06d}",
                       "timestamp_ns": SYNTHETIC_PROBE_START_NS + round_index * SYNTHETIC_ROUND_INTERVAL_NS,
                       "observations": observations})
    for prefill_round in range(config.prefill_rounds):
        append_round("prefill", prefill_round, 0)
    boundaries = [{"timestamp_ns": SYNTHETIC_REFERENCE_START_NS, "phase": "prefill", "step_id": None, "step_index": None}]
    for step_index, step_id in enumerate(ids["step_ids"]):
        boundaries.append({"timestamp_ns": SYNTHETIC_REFERENCE_START_NS + len(rounds) * SYNTHETIC_ROUND_INTERVAL_NS,
                           "phase": "decoding", "step_id": step_id, "step_index": step_index})
        aggregate = sum(integer_kv_aggregate(_query_accesses(config, index, "decoding", step_index, *position, split), kv_values)
                        for position in head_positions.values())
        gold_steps.append({"step_id": step_id, "step_index": step_index,
                           "gold_token_id": toy_token_id_from_aggregate(aggregate)})
        for vote in range(config.decode_rounds):
            append_round("decoding", step_index, vote)
    return {
        "schema_version": "janus.probe.run.v1", "run_id": ids["run_id"],
        "source_kind": "synthetic_probe_simulation", "provenance": _simulation_provenance(config),
        "task_id": "cpu-toy-integer-kv-sign", "case_id": ids["case_id"], "sample_id": ids["sample_id"],
        "response_id": ids["response_id"], "split": split, "split_assignment_id": assignment_id,
        "allocation_epoch": layout["allocation_epoch"],
        "clock": {"unit": "ns", "probe_collection_start_ns": SYNTHETIC_PROBE_START_NS,
                  "reference_collection_start_ns": SYNTHETIC_REFERENCE_START_NS},
        "layout": layout, "calibration": calibration, "probe_rounds": rounds,
        "phase_reference": {"source_kind": "oracle_annotation", "reference_method": "cpu_toy_execution_boundary_reference_v1",
                            "alignment_evidence_id": f"synthetic-boundary-reference:{ids['run_id']}",
                            "boundaries": boundaries,
                            "collection_end_ns": SYNTHETIC_REFERENCE_START_NS + len(rounds) * SYNTHETIC_ROUND_INTERVAL_NS},
        "gold": {"source_kind": "oracle_annotation",
                 "oracle_method": "actual_integer_kv_gather_sum_sign;test_last_step_balanced_zero_sum_is_declared_OOV23;symbolic_table_frozen_before_generation",
                 "attributes": {"toy_topic": "alpha" if index % 2 == 0 else "beta"},
                 "tokenizer": {key: _TOKENIZER[key] for key in ("name", "revision", "vocab_size")},
                 "alignment": {"step_index_base": 0, "profile_predicts": "same_index_output_token",
                               "bos_included": False, "eos_included": False, "special_tokens": "excluded",
                               "response_scope": "complete_response"},
                 "steps": gold_steps},
    }


def generate_controlled_workload(config: WorkloadConfig) -> dict[str, Any]:
    if not isinstance(config, WorkloadConfig):
        raise ValidationError("generate_controlled_workload requires WorkloadConfig")
    config.validate()
    split_contract = _split_contract(config)
    # The complete assignment is immutable before the first query/probe/gold generation.
    runs = [_generate_run(index, assignment["split"], split_contract["split_assignment_id"], config)
            for index, assignment in enumerate(split_contract["assignments"])]
    bundle = {"schema_version": WORKLOAD_SCHEMA_VERSION, "config": asdict(config),
              "split_contract": split_contract, "tokenizer": toy_tokenizer_contract(), "runs": runs}
    return validate_controlled_workload(bundle)


def validate_controlled_workload(bundle: Any) -> dict[str, Any]:
    """Validate the fixed synthetic wrapper and original case assignment evidence.

    Raw JSON loaders validate individual probe runs. This wrapper additionally
    binds config, token table, fixed IDs and assignments to the bundle contract.
    It verifies simulated raw values from config independently of oracle gold.
    Neither digests nor oracle fields authenticate physical recordings.
    """
    if not isinstance(bundle, dict) or set(bundle) != _BUNDLE_FIELDS:
        raise ValidationError(f"controlled workload fields must equal {sorted(_BUNDLE_FIELDS)}")
    if bundle["schema_version"] != WORKLOAD_SCHEMA_VERSION:
        raise ValidationError("invalid controlled workload schema_version")
    config = WorkloadConfig.from_dict(bundle["config"])
    split_contract = _split_contract(config)
    if canonical_sha256(bundle["split_contract"]) != canonical_sha256(split_contract):
        raise ValidationError("controlled workload fixed split contract/digest mismatch")
    if canonical_sha256(bundle["tokenizer"]) != canonical_sha256(_TOKENIZER):
        raise ValidationError("controlled workload fixed tokenizer contract mismatch")
    if not isinstance(bundle["runs"], list) or len(bundle["runs"]) != config.case_count:
        raise ValidationError("controlled workload run count does not match config.case_count")
    runs = validate_probe_runs(bundle["runs"])
    run_by_case = {run["case_id"]: run for run in runs}
    if len(run_by_case) != config.case_count or set(run_by_case) != {row["case_id"] for row in split_contract["assignments"]}:
        raise ValidationError("controlled workload requires exactly one run for each fixed case ID")
    expected_round_count = config.prefill_rounds + config.steps_per_response * config.decode_rounds
    for index, assignment in enumerate(split_contract["assignments"]):
        run = run_by_case[assignment["case_id"]]
        ids = _ids(index, config)
        if run["source_kind"] != "synthetic_probe_simulation" or canonical_sha256(run["provenance"]) != canonical_sha256(_simulation_provenance(config)):
            raise ValidationError("controlled workload requires its declared synthetic simulator/config provenance")
        if run["task_id"] != "cpu-toy-integer-kv-sign" or any(run[key] != ids[key] for key in ("run_id", "sample_id", "response_id")):
            raise ValidationError("controlled workload stable task/run/sample/response ID mismatch")
        if run["split"] != assignment["split"] or run["split_assignment_id"] != split_contract["split_assignment_id"]:
            raise ValidationError("controlled workload run split differs from its frozen case assignment")
        expected_layout, expected_calibration = _layout(index, config)
        if run["layout"] != expected_layout or run["calibration"] != expected_calibration:
            raise ValidationError("controlled workload synthetic geometry/calibration differs from config")
        if run["clock"] != {"unit": "ns", "probe_collection_start_ns": SYNTHETIC_PROBE_START_NS,
                            "reference_collection_start_ns": SYNTHETIC_REFERENCE_START_NS}:
            raise ValidationError("controlled workload clock differs from the named synthetic timing policy")
        for round_index, probe_round in enumerate(run["probe_rounds"]):
            if probe_round["round_id"] != f"{ids['run_id']}:probe-round-{round_index:06d}" or probe_round["timestamp_ns"] != SYNTHETIC_PROBE_START_NS + round_index * SYNTHETIC_ROUND_INTERVAL_NS:
                raise ValidationError("controlled workload raw round timing/IDs differ from config")
            if round_index < config.prefill_rounds:
                phase, query_index, vote_index = "prefill", round_index, 0
            else:
                decode_index = round_index - config.prefill_rounds
                phase, query_index, vote_index = "decoding", decode_index // config.decode_rounds, decode_index % config.decode_rounds
            expected_observations = _expected_observations(config, index, assignment["split"], ids,
                                                           expected_layout, phase, query_index, vote_index)
            if {observation["probe_id"]: observation for observation in probe_round["observations"]} != {observation["probe_id"]: observation for observation in expected_observations}:
                raise ValidationError("controlled workload raw observations differ from declared deterministic simulator/config")
        expected_boundaries = [{"timestamp_ns": SYNTHETIC_REFERENCE_START_NS, "phase": "prefill", "step_id": None, "step_index": None}]
        expected_boundaries.extend({"timestamp_ns": SYNTHETIC_REFERENCE_START_NS + (config.prefill_rounds + step_index * config.decode_rounds) * SYNTHETIC_ROUND_INTERVAL_NS,
                                    "phase": "decoding", "step_id": step_id, "step_index": step_index}
                                   for step_index, step_id in enumerate(ids["step_ids"]))
        reference = run["phase_reference"]
        if reference["reference_method"] != "cpu_toy_execution_boundary_reference_v1" or reference["alignment_evidence_id"] != f"synthetic-boundary-reference:{ids['run_id']}" or reference["boundaries"] != expected_boundaries or reference["collection_end_ns"] != SYNTHETIC_REFERENCE_START_NS + expected_round_count * SYNTHETIC_ROUND_INTERVAL_NS:
            raise ValidationError("controlled workload oracle boundary reference differs from config")
        if run["gold"]["oracle_method"] != "actual_integer_kv_gather_sum_sign;test_last_step_balanced_zero_sum_is_declared_OOV23;symbolic_table_frozen_before_generation" or run["gold"]["alignment"] != {"step_index_base": 0, "profile_predicts": "same_index_output_token", "bos_included": False, "eos_included": False, "special_tokens": "excluded", "response_scope": "complete_response"}:
            raise ValidationError("controlled workload oracle/token alignment policy differs from config")
        if len(run["probe_rounds"]) != expected_round_count:
            raise ValidationError("controlled workload round count differs from config")
        if [step["step_id"] for step in run["gold"]["steps"]] != ids["step_ids"]:
            raise ValidationError("controlled workload stable step IDs differ from config")
        if run["gold"]["tokenizer"] != {key: _TOKENIZER[key] for key in ("name", "revision", "vocab_size")}:
            raise ValidationError("controlled workload gold tokenizer mismatch")
        # Oracle values are a sidecar: validate their declared token interface,
        # but do not regenerate labels from raw features or overwrite annotations.
        # This lets the reconstruction boundary demonstrate gold independence.
        if any(step["gold_token_id"] not in {11, 17, 23} for step in run["gold"]["steps"]):
            raise ValidationError("controlled workload gold token is outside its fixed symbolic table")
    result = {"schema_version": WORKLOAD_SCHEMA_VERSION, "config": asdict(config),
              "split_contract": split_contract, "tokenizer": toy_tokenizer_contract(),
              "runs": [run_by_case[row["case_id"]] for row in split_contract["assignments"]]}
    return deepcopy(result)


def load_controlled_workload(path: str | Path) -> dict[str, Any]:
    return validate_controlled_workload(_load_bounded_json(path))


def write_controlled_workload(bundle: dict[str, Any], path: str | Path) -> Path:
    canonical = validate_controlled_workload(bundle)
    encoded = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) + 1 > MAX_WORKLOAD_FILE_BYTES:
        raise ValidationError("controlled workload serialized file byte budget exceeded")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded + b"\n")
    return destination
