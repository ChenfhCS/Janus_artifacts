"""Offline, reference-aligned reconstruction of declared logical page probes.

The phase reference is an oracle annotation.  Page-to-token distributions are
explicit numerical reconstruction choices, not recovered physical addresses or
the victim's true token activations.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from .probe_contract import MAX_INT, MIN_INT, canonical_sha256, validate_probe_run
from .schema import ValidationError


RECONSTRUCTION_REVISION = "janus.probe.reconstruction.v1"
MAX_CONFIG_BYTES = 64 * 1024
MAX_VOTES = 1_000_000


def _finite_nonnegative(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValidationError(f"{name} must be a finite numeric value")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be finite") from exc
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValidationError(f"{name} must be finite and {qualifier}")
    return number


@dataclass(frozen=True)
class ReconstructionConfig:
    """Explicit numerical policies used by this offline reconstruction."""

    max_alignment_error_ns: float = 5
    alignment_tie_policy: str = "earlier"
    min_decode_votes: int = 3
    majority_tie_policy: str = "reject"
    prefill_max_exponent: float = 2
    decoding_density_radius: int = 1
    decoding_placement: str = "prefix"
    density_rounding: str = "ceil"

    def validate(self) -> None:
        _finite_nonnegative(self.max_alignment_error_ns, "max_alignment_error_ns")
        _finite_nonnegative(self.prefill_max_exponent, "prefill_max_exponent")
        if type(self.alignment_tie_policy) is not str or self.alignment_tie_policy not in {"earlier", "later", "reject"}:
            raise ValidationError("alignment_tie_policy must be earlier, later, or reject")
        if type(self.majority_tie_policy) is not str or self.majority_tie_policy != "reject":
            raise ValidationError("majority_tie_policy must be reject")
        if (
            type(self.min_decode_votes) is not int
            or self.min_decode_votes <= 0
        ):
            raise ValidationError("min_decode_votes must be a positive integer")
        if (
            type(self.decoding_density_radius) is not int
            or self.decoding_density_radius < 0
        ):
            raise ValidationError("decoding_density_radius must be a nonnegative integer")
        if type(self.decoding_placement) is not str or self.decoding_placement != "prefix":
            raise ValidationError("decoding_placement must be prefix")
        if type(self.density_rounding) is not str or self.density_rounding != "ceil":
            raise ValidationError("density_rounding must be ceil")

    @classmethod
    def from_json(cls, path: str | Path) -> "ReconstructionConfig":
        source = Path(path)
        try:
            if source.stat().st_size > MAX_CONFIG_BYTES:
                raise ValidationError("reconstruction config exceeds the file-size limit")

            def unique_object(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValidationError(f"duplicate reconstruction config field: {key}")
                    result[key] = value
                return result

            raw = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        except ValidationError:
            raise
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            raise ValidationError("invalid reconstruction config JSON") from exc
        expected = set(cls.__dataclass_fields__)
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValidationError(
                f"reconstruction config fields must equal {sorted(expected)}"
            )
        config = cls(**raw)
        config.validate()
        return config


def kernel_utilization_proxy(
    cumulative_kernel_duration_ns: float, window_duration_ns: float
) -> float:
    """Return cumulative kernel duration / window duration, without clamping.

    Summed concurrent kernel durations can exceed the window.  This is a
    timeline-derived proxy, not a GPU hardware utilization measurement.
    """
    cumulative = _finite_nonnegative(
        cumulative_kernel_duration_ns, "cumulative_kernel_duration_ns"
    )
    window = _finite_nonnegative(window_duration_ns, "window_duration_ns", positive=True)
    result = cumulative / window
    if not math.isfinite(result):
        raise ValidationError("kernel utilization proxy is not finite")
    return result


def threshold_page_event(latency: float, threshold: float) -> int:
    """Equality is inactive: translation reload latency must exceed threshold."""
    _finite_nonnegative(latency, "reload_latency_ns")
    _finite_nonnegative(threshold, "translation_threshold_ns")
    # Preserve integer ordering even above the exact-integer range of float64.
    return int(latency > threshold)


def majority_vote(votes: Iterable[int], tie_policy: str = "reject") -> int:
    """Aggregate explicitly supplied binary votes; ambiguous ties are rejected."""
    if type(tie_policy) is not str or tie_policy != "reject":
        raise ValidationError("majority tie_policy must be reject")
    total = 0
    active = 0
    try:
        for vote in votes:
            if type(vote) is not int or vote not in {0, 1}:
                raise ValidationError("majority votes must be binary integers")
            total += 1
            if total > MAX_VOTES:
                raise ValidationError("majority votes exceed the observation budget")
            active += vote
    except TypeError as exc:
        raise ValidationError("majority votes must be an iterable of binary integers") from exc
    if total == 0:
        raise ValidationError("majority votes must not be empty")
    if 2 * active == total:
        raise ValidationError("majority vote tie is ambiguous")
    return int(2 * active > total)


def _exact_time(value: int | float) -> Fraction:
    """Preserve an integer or the exact value of an already validated IEEE float."""
    return Fraction(value) if type(value) is int else Fraction(*value.as_integer_ratio())


def _exact_json_number(value: Fraction, name: str) -> int | float:
    """Emit an exact scalar under the existing primitive JSON number budget.

    A dyadic result can still exceed float precision. Reject such results instead
    of recording a rounded offset/error that contradicts the alignment decision.
    """
    if value.denominator == 1 and MIN_INT <= value.numerator <= MAX_INT:
        return value.numerator
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValidationError(
            f"{name} has unsupported precision/range for an exact finite JSON number"
        ) from exc
    if not math.isfinite(number) or _exact_time(number) != value:
        raise ValidationError(
            f"{name} has unsupported precision/range for an exact finite JSON number"
        )
    return number


def _nearest_round_index(
    timestamps: list[Fraction], boundary: Fraction, tie_policy: str
) -> tuple[int, Fraction]:
    position = bisect.bisect_left(timestamps, boundary)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(timestamps)]
    distances = {index: abs(timestamps[index] - boundary) for index in candidates}
    distance = min(distances.values())
    nearest = [index for index in candidates if distances[index] == distance]
    if len(nearest) > 1:
        if tie_policy == "reject":
            raise ValidationError("reference boundary has an ambiguous nearest-round tie")
        selected = min(nearest) if tie_policy == "earlier" else max(nearest)
    else:
        selected = nearest[0]
    return selected, distance


def _segment_validated(
    run: dict[str, Any], config: ReconstructionConfig
) -> tuple[list[dict[str, Any]], int | float]:
    rounds = run["probe_rounds"]
    timestamps = [_exact_time(round_record["timestamp_ns"]) for round_record in rounds]
    reference = run["phase_reference"]
    offset = (
        _exact_time(run["clock"]["probe_collection_start_ns"])
        - _exact_time(run["clock"]["reference_collection_start_ns"])
    )
    serialized_offset = _exact_json_number(offset, "clock_alignment_offset_ns")
    tolerance = _exact_time(config.max_alignment_error_ns)
    end = _exact_time(reference["collection_end_ns"]) + offset
    if end <= timestamps[-1]:
        raise ValidationError("aligned collection end must cover every probe round")
    boundaries = []
    for boundary in reference["boundaries"]:
        aligned = _exact_time(boundary["timestamp_ns"]) + offset
        index, error = _nearest_round_index(
            timestamps, aligned, config.alignment_tie_policy
        )
        if error > tolerance:
            raise ValidationError("reference boundary alignment exceeds max_alignment_error_ns")
        if boundaries and index <= boundaries[-1][0]:
            raise ValidationError("reference boundaries collapse onto the same probe round")
        boundaries.append((index, error, boundary))
    if boundaries[0][0] != 0:
        raise ValidationError("reference alignment would omit leading probe rounds")
    segments = []
    for position, (start, error, boundary) in enumerate(boundaries):
        stop = boundaries[position + 1][0] if position + 1 < len(boundaries) else len(rounds)
        selected = rounds[start:stop]
        if not selected:
            raise ValidationError("reference-aligned phase has no probe rounds")
        if boundary["phase"] == "decoding" and len(selected) < config.min_decode_votes:
            raise ValidationError("reference-aligned decoding step has insufficient probe votes")
        segments.append({
            "phase": boundary["phase"],
            "step_id": boundary["step_id"],
            "step_index": boundary["step_index"],
            "round_ids": [round_record["round_id"] for round_record in selected],
            "alignment_error_ns": _exact_json_number(error, "alignment_error_ns"),
            "source_kind": "oracle_aligned_phase",
        })
    return segments, serialized_offset


def segment_probe_run(
    run: dict[str, Any], config: ReconstructionConfig
) -> list[dict[str, Any]]:
    """Snap declared oracle/reference boundaries to a fully covered probe timeline."""
    if not isinstance(config, ReconstructionConfig):
        raise ValidationError("config must be a ReconstructionConfig")
    config.validate()
    return _segment_validated(validate_probe_run(run), config)[0]


def _blank_profile(layout: dict[str, Any]) -> list[list[list[float]]]:
    return [
        [[0.0] * layout["token_width"] for _ in layout["kv_head_ids"]]
        for _ in layout["layer_ids"]
    ]


def reconstruct_probe_run(
    run: dict[str, Any], config: ReconstructionConfig
) -> dict[str, Any]:
    """Reconstruct proxy/representative profiles from calibrated logical mappings."""
    if not isinstance(config, ReconstructionConfig):
        raise ValidationError("config must be a ReconstructionConfig")
    config.validate()
    validated = validate_probe_run(run)
    segments, clock_alignment_offset = _segment_validated(validated, config)
    layout = validated["layout"]
    layer_indices = {name: index for index, name in enumerate(layout["layer_ids"])}
    head_indices = {name: index for index, name in enumerate(layout["kv_head_ids"])}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for page in layout["page_entries"]:
        groups.setdefault((page["layer_id"], page["kv_head_id"]), []).append(page)
    for pages in groups.values():
        pages.sort(key=lambda page: page["page_order"])
    calibration = {
        entry["probe_id"]: entry for entry in validated["calibration"]["probes"]
    }
    rounds = {
        entry["round_id"]: {
            observation["probe_id"]: observation for observation in entry["observations"]
        }
        for entry in validated["probe_rounds"]
    }
    prefill_segment = segments[0]
    prefill_profile = _blank_profile(layout)
    prefill_events = []
    for (layer_id, head_id), pages in groups.items():
        counts = [
            sum(
                rounds[round_id][page["probe_id"]]["evicted_probe_lines"]
                for round_id in prefill_segment["round_ids"]
            )
            for page in pages
        ]
        maximum = max(counts)
        for page, count in zip(pages, counts):
            frequency = _finite_nonnegative(count, "relative_frequency_proxy")
            alpha = (
                config.prefill_max_exponent * (count / maximum) if maximum else 0.0
            )
            # rank 1 always has weight 1, so normalization is positive even for
            # steep concentration; exp avoids overflow from positive powers.
            weights = [
                math.exp(-alpha * math.log(rank + 1))
                for rank in range(page["token_count"])
            ]
            denominator = sum(weights)
            token_weights = [frequency * (weight / denominator) for weight in weights]
            if not all(math.isfinite(value) for value in token_weights):
                raise ValidationError("prefill reconstruction produced non-finite values")
            start = page["token_start"]
            profile = prefill_profile[layer_indices[layer_id]][head_indices[head_id]]
            profile[start:start + page["token_count"]] = token_weights
            prefill_events.append({
                **{key: page[key] for key in (
                    "page_id", "probe_id", "layer_id", "kv_head_id", "page_order"
                )},
                "relative_frequency_proxy": count,
            })

    decoding_profiles = []
    decoding_events = []
    for segment in segments[1:]:
        profile = _blank_profile(layout)
        step_pages = []
        for (layer_id, head_id), pages in groups.items():
            page_votes = [
                [
                    threshold_page_event(
                        rounds[round_id][page["probe_id"]]["reload_latency_ns"],
                        calibration[page["probe_id"]]["translation_threshold_ns"],
                    )
                    for round_id in segment["round_ids"]
                ]
                for page in pages
            ]
            events = [
                majority_vote(votes, tie_policy=config.majority_tie_policy)
                for votes in page_votes
            ]
            for index, (page, votes, event) in enumerate(zip(pages, page_votes, events)):
                left = max(0, index - config.decoding_density_radius)
                right = min(len(pages), index + config.decoding_density_radius + 1)
                neighbor_count = right - left
                active_neighbors = sum(events[left:right])
                density = active_neighbors / neighbor_count
                # Exact integer ceil of density * token_count avoids accidental
                # rounding down at a floating point integer boundary.
                representative_count = (
                    max(1, min(
                        page["token_count"],
                        (active_neighbors * page["token_count"] + neighbor_count - 1)
                        // neighbor_count,
                    ))
                    if event else 0
                )
                start = page["token_start"]
                profile[layer_indices[layer_id]][head_indices[head_id]][
                    start:start + representative_count
                ] = [1.0] * representative_count
                step_pages.append({
                    **{key: page[key] for key in (
                        "page_id", "probe_id", "layer_id", "kv_head_id", "page_order"
                    )},
                    "votes": votes,
                    "event": event,
                    "density": density,
                    "active_token_count": representative_count,
                })
        step_identity = {
            "step_id": segment["step_id"], "step_index": segment["step_index"]
        }
        decoding_profiles.append({**step_identity, "profile": profile})
        decoding_events.append({**step_identity, "pages": step_pages})

    return {
        **{key: validated[key] for key in (
            "run_id", "sample_id", "case_id", "response_id", "split"
        )},
        "source_kind": "reconstructed_probe_profile",
        "scientific_result": False,
        "phase_source_kind": "oracle_aligned_phase",
        "prefill_profile": prefill_profile,
        "decoding_steps": decoding_profiles,
        "page_events": {"prefill": prefill_events, "decoding": decoding_events},
        "provenance": {
            "source_kind": "offline_probe_reconstruction",
            "reconstruction_revision": RECONSTRUCTION_REVISION,
            "config": asdict(config),
            "input_sha256": canonical_sha256(validated),
            "input_source_kind": validated["source_kind"],
            "calibration_id": validated["calibration"]["calibration_id"],
            "allocation_epoch": validated["allocation_epoch"],
            "reference_method": validated["phase_reference"]["reference_method"],
            "alignment_evidence_id": validated["phase_reference"]["alignment_evidence_id"],
            "clock_alignment_offset_ns": clock_alignment_offset,
            "phase_segments": segments,
            "axis_contract": ["layer", "kv_head", "token"],
            "logical_mapping_source_kind": layout["source_kind"],
            "prefill_frequency_interpretation": "relative_eviction_frequency_proxy",
            "prefill_distribution": "page_frequency_normalized_power_law_by_token_index",
            "decoding_interpretation": "representative_token_sparsity_not_true_activations",
            "decoding_density": "clipped_uniform_neighborhood_within_layer_and_head",
            "physical_addresses_recovered": False,
            "trace_only_phase_detection": False,
        },
    }
