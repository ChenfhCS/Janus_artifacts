"""CPU-only synthetic regression tests for declared probe reconstruction."""

import copy
import json
import math
import tempfile
import unittest
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path

from artifacts.janus_artifact.probe_reconstruction import (
    ReconstructionConfig,
    kernel_utilization_proxy,
    majority_vote,
    reconstruct_probe_run,
    segment_probe_run,
    threshold_page_event,
)
from artifacts.janus_artifact.probe_contract import canonical_sha256
from artifacts.janus_artifact.schema import ValidationError


ARTIFACTS_ROOT = Path(__file__).resolve().parents[1]


def make_probe_run():
    """A declared toy mapping with noisy positives and noisy negatives."""
    pages = []
    probes = []
    for head in range(2):
        for page in range(3):
            probe_id = f"probe-{head}-{page}"
            pages.append({
                "page_id": f"page-{head}-{page}",
                "probe_id": probe_id,
                "layer_id": "layer-0",
                "kv_head_id": f"head-{head}",
                "page_order": page,
                "token_start": page * 4,
                "token_count": 4,
            })
            probes.append({
                "probe_id": probe_id,
                "translation_threshold_ns": 150,
                "probe_set_line_count": 8,
            })
    timestamps = [100, 101, 102, 110, 111, 112, 113, 114]
    rounds = []
    for index, timestamp in enumerate(timestamps):
        observations = []
        for head in range(2):
            for page in range(3):
                if index < 3:
                    evictions = (4, 2, 0)[page] if head == 0 else 1
                    latency = 100
                else:
                    evictions = 0
                    step_round = index - 3
                    if head == 0 and page == 0:
                        active = [1, 1, 0, 1, 1][step_round]
                    elif head == 0 and page == 1:
                        active = 1
                    elif head == 0 and page == 2:
                        active = [0, 1, 0, 0, 0][step_round]
                    else:
                        active = 0
                    latency = 200 if active else 100
                observations.append({
                    "probe_id": f"probe-{head}-{page}",
                    "reload_latency_ns": latency,
                    "evicted_probe_lines": evictions,
                })
        rounds.append({
            "round_id": f"round-{index}",
            "timestamp_ns": timestamp,
            "observations": observations,
        })
    return {
        "schema_version": "janus.probe.run.v1",
        "run_id": "run-0",
        "source_kind": "synthetic_probe_simulation",
        "provenance": {"synthetic": True, "simulator_revision": "toy-fixture-v1"},
        "task_id": "task-0",
        "case_id": "case-0",
        "sample_id": "sample-0",
        "response_id": "response-0",
        "split": "train",
        "split_assignment_id": "assignment-0",
        "allocation_epoch": "epoch-0",
        "clock": {
            "unit": "ns",
            "probe_collection_start_ns": 100,
            "reference_collection_start_ns": 0,
        },
        "layout": {
            "source_kind": "synthetic_layout",
            "calibration_id": "calibration-0",
            "allocation_epoch": "epoch-0",
            "layer_ids": ["layer-0"],
            "kv_head_ids": ["head-0", "head-1"],
            "token_width": 12,
            "page_entries": pages,
        },
        "calibration": {
            "source_kind": "synthetic_calibration",
            "calibration_id": "calibration-0",
            "allocation_epoch": "epoch-0",
            "probes": probes,
        },
        "probe_rounds": rounds,
        "phase_reference": {
            "source_kind": "oracle_annotation",
            "reference_method": "toy-schedule-boundaries",
            "alignment_evidence_id": "alignment-0",
            "boundaries": [
                {"timestamp_ns": 0, "phase": "prefill", "step_id": None, "step_index": None},
                {"timestamp_ns": 10, "phase": "decoding", "step_id": "step-0", "step_index": 0},
            ],
            "collection_end_ns": 15,
        },
        "gold": {
            "source_kind": "oracle_annotation",
            "oracle_method": "toy-output-interface",
            "attributes": {"toy_topic": "alpha"},
            "tokenizer": {"name": "toy-symbols", "revision": "toy-v1", "vocab_size": 24},
            "alignment": {
                "step_index_base": 0,
                "profile_predicts": "same_index_output_token",
                "bos_included": False,
                "eos_included": False,
                "special_tokens": "excluded",
                "response_scope": "complete_response",
            },
            "steps": [{"step_id": "step-0", "step_index": 0, "gold_token_id": 11}],
        },
    }


def make_precision_probe_run(epoch=0):
    """Represent large absolute epochs as integers and fractional offsets as floats."""
    run = make_probe_run()
    run["clock"]["probe_collection_start_ns"] = epoch
    run["clock"]["reference_collection_start_ns"] = 0.0
    for record, delta in zip(
        run["probe_rounds"], [100, 101, 102, 1024, 1025, 1026, 1027, 1028]
    ):
        record["timestamp_ns"] = epoch + delta
    run["phase_reference"]["boundaries"][0]["timestamp_ns"] = 100.0
    run["phase_reference"]["boundaries"][1]["timestamp_ns"] = 1024.0
    run["phase_reference"]["collection_end_ns"] = 2048.0
    return run


class ProbeReconstructionTests(unittest.TestCase):
    def test_kernel_proxy_allows_concurrent_sum_above_one(self):
        self.assertEqual(kernel_utilization_proxy(25, 10), 2.5)
        self.assertEqual(kernel_utilization_proxy(0, 10), 0.0)
        for cumulative, window in (
            (-1, 10), (1, 0), (1, -1), (True, 10),
            (1, False), (float("nan"), 10), (1, float("inf")),
            (1e308, 1e-308), ("1", 10),
        ):
            with self.subTest(cumulative=cumulative, window=window):
                with self.assertRaises(ValidationError):
                    kernel_utilization_proxy(cumulative, window)

    def test_latency_threshold_is_strict_and_preserves_integer_ordering(self):
        self.assertEqual(threshold_page_event(149, 150), 0)
        self.assertEqual(threshold_page_event(150, 150), 0)
        self.assertEqual(threshold_page_event(151, 150), 1)
        self.assertEqual(threshold_page_event(2**60 + 1, 2**60), 1)
        for latency, threshold in (
            (-1, 0), (1, -1), (float("nan"), 1), (1, float("inf")),
            (True, 1), (1, False),
        ):
            with self.subTest(latency=latency, threshold=threshold):
                with self.assertRaises(ValidationError):
                    threshold_page_event(latency, threshold)

    def test_noisy_majority_positive_negative_and_ambiguous_inputs(self):
        self.assertEqual(majority_vote([1, 0, 1, 1, 0]), 1)
        self.assertEqual(majority_vote([0, 1, 0, 0, 1]), 0)
        for votes in ([], [1, 0], [1, 0, 1, 0], [True, 1], [1.0], [2], None):
            with self.subTest(votes=votes):
                with self.assertRaises(ValidationError):
                    majority_vote(votes)
        with self.assertRaisesRegex(ValidationError, "tie_policy"):
            majority_vote([1], tie_policy="earlier")

    def test_config_rejects_invalid_types_nonfinite_and_unimplemented_policies(self):
        for field, value in (
            ("max_alignment_error_ns", float("nan")),
            ("max_alignment_error_ns", True),
            ("max_alignment_error_ns", -1),
            ("prefill_max_exponent", float("inf")),
            ("prefill_max_exponent", -1),
            ("alignment_tie_policy", "random"),
            ("alignment_tie_policy", []),
            ("min_decode_votes", 0),
            ("min_decode_votes", True),
            ("decoding_density_radius", -1),
            ("decoding_density_radius", True),
            ("majority_tie_policy", "zero"),
            ("decoding_placement", "random"),
            ("density_rounding", "floor"),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    replace(ReconstructionConfig(), **{field: value}).validate()

    def test_config_json_requires_exact_fields_and_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory(
            prefix=".probe-config-test-", dir=ARTIFACTS_ROOT
        ) as directory:
            path = Path(directory) / "config.json"
            expected = ReconstructionConfig()
            path.write_text(json.dumps(asdict(expected)), encoding="utf-8")
            self.assertEqual(ReconstructionConfig.from_json(path), expected)
            for raw in (
                {},
                {**asdict(expected), "unknown": 1},
                {**asdict(expected), "min_decode_votes": 3.0},
                {**asdict(expected), "prefill_max_exponent": float("nan")},
            ):
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaises(ValidationError):
                    ReconstructionConfig.from_json(path)
            path.write_text('{"min_decode_votes":3,"min_decode_votes":5}', encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "duplicate"):
                ReconstructionConfig.from_json(path)
            for malformed in ("[" * 2000 + "0" + "]" * 2000, "9" * 5000):
                path.write_text(malformed, encoding="utf-8")
                with self.assertRaises(ValidationError):
                    ReconstructionConfig.from_json(path)

    def test_reference_clock_alignment_preserves_all_rounds_and_explicit_source(self):
        run = make_probe_run()
        segments = segment_probe_run(run, ReconstructionConfig())
        self.assertEqual(segments[0]["round_ids"], ["round-0", "round-1", "round-2"])
        self.assertEqual(segments[1]["round_ids"], [f"round-{index}" for index in range(3, 8)])
        self.assertTrue(all(segment["source_kind"] == "oracle_aligned_phase" for segment in segments))
        self.assertTrue(all(segment["alignment_error_ns"] == 0 for segment in segments))
        shifted = copy.deepcopy(run)
        shifted["clock"]["reference_collection_start_ns"] = 1000
        for boundary in shifted["phase_reference"]["boundaries"]:
            boundary["timestamp_ns"] += 1000
        shifted["phase_reference"]["collection_end_ns"] += 1000
        self.assertEqual(segment_probe_run(shifted, ReconstructionConfig()), segments)

    def test_nearest_round_tie_has_explicit_earlier_later_or_reject_policy(self):
        run = make_probe_run()
        run["phase_reference"]["boundaries"][1]["timestamp_ns"] = 10.5
        earlier = segment_probe_run(run, ReconstructionConfig(alignment_tie_policy="earlier"))
        later = segment_probe_run(run, ReconstructionConfig(alignment_tie_policy="later"))
        self.assertEqual(earlier[1]["round_ids"][0], "round-3")
        self.assertEqual(later[1]["round_ids"][0], "round-4")
        self.assertEqual(earlier[1]["alignment_error_ns"], 0.5)
        self.assertEqual(later[1]["alignment_error_ns"], 0.5)
        with self.assertRaisesRegex(ValidationError, "tie"):
            segment_probe_run(run, ReconstructionConfig(alignment_tie_policy="reject"))

    def test_skew_collapse_incomplete_coverage_and_insufficient_votes_are_rejected(self):
        skew = make_probe_run()
        skew["phase_reference"]["boundaries"][1]["timestamp_ns"] = 6
        with self.assertRaisesRegex(ValidationError, "alignment"):
            segment_probe_run(skew, ReconstructionConfig(max_alignment_error_ns=3))
        collapsed = make_probe_run()
        collapsed["phase_reference"]["boundaries"][1]["timestamp_ns"] = 0.4
        with self.assertRaisesRegex(ValidationError, "collapse"):
            segment_probe_run(collapsed, ReconstructionConfig())
        omitted = make_probe_run()
        omitted["phase_reference"]["boundaries"][0]["timestamp_ns"] = 1
        with self.assertRaisesRegex(ValidationError, "leading"):
            segment_probe_run(omitted, ReconstructionConfig())
        uncovered = make_probe_run()
        uncovered["phase_reference"]["collection_end_ns"] = 14
        with self.assertRaisesRegex(ValidationError, "cover"):
            segment_probe_run(uncovered, ReconstructionConfig())
        with self.assertRaisesRegex(ValidationError, "insufficient"):
            segment_probe_run(make_probe_run(), ReconstructionConfig(min_decode_votes=6))

    def test_large_epoch_real_skew_rejected_at_zero_and_default_tolerance(self):
        run = make_precision_probe_run(2**60)
        run["clock"]["reference_collection_start_ns"] = 0
        run["phase_reference"]["boundaries"][0]["timestamp_ns"] = 0.0
        for config in (ReconstructionConfig(max_alignment_error_ns=0), ReconstructionConfig()):
            with self.subTest(tolerance=config.max_alignment_error_ns):
                with self.assertRaisesRegex(ValidationError, "alignment"):
                    segment_probe_run(run, config)
                with self.assertRaisesRegex(ValidationError, "alignment"):
                    reconstruct_probe_run(run, config)
        segments = segment_probe_run(run, ReconstructionConfig(max_alignment_error_ns=100))
        self.assertEqual(segments[0]["alignment_error_ns"], 100)
        self.assertIs(type(segments[0]["alignment_error_ns"]), int)

    def test_mixed_integral_float_epochs_preserve_segmentation_and_exact_offset(self):
        baseline = make_precision_probe_run()
        baseline["clock"]["probe_collection_start_ns"] = 1
        for record in baseline["probe_rounds"]:
            record["timestamp_ns"] += 1
        shifted = copy.deepcopy(baseline)
        epoch = 2**60
        shifted["clock"]["probe_collection_start_ns"] += epoch
        shifted["clock"]["reference_collection_start_ns"] = float(epoch)
        for record in shifted["probe_rounds"]:
            record["timestamp_ns"] += epoch
        for boundary in shifted["phase_reference"]["boundaries"]:
            boundary["timestamp_ns"] = epoch + int(boundary["timestamp_ns"])
        shifted["phase_reference"]["collection_end_ns"] = epoch + int(
            shifted["phase_reference"]["collection_end_ns"]
        )
        config = ReconstructionConfig(max_alignment_error_ns=0)
        first = reconstruct_probe_run(baseline, config)
        second = reconstruct_probe_run(shifted, config)
        self.assertEqual(first["provenance"]["phase_segments"], second["provenance"]["phase_segments"])
        self.assertEqual(first["prefill_profile"], second["prefill_profile"])
        self.assertEqual(first["decoding_steps"], second["decoding_steps"])
        self.assertEqual(first["provenance"]["clock_alignment_offset_ns"], 1)
        self.assertEqual(second["provenance"]["clock_alignment_offset_ns"], 1)
        for epoch_shift in (0, epoch):
            run = make_precision_probe_run(epoch_shift)
            run["clock"]["probe_collection_start_ns"] = float(epoch_shift)
            result = reconstruct_probe_run(run, config)
            self.assertEqual(result["provenance"]["phase_segments"], first["provenance"]["phase_segments"])
            self.assertEqual(result["provenance"]["clock_alignment_offset_ns"], epoch_shift)
            self.assertEqual(json.loads(json.dumps(result)), result)
            canonical_sha256(result)

    def test_large_epoch_true_fractional_nearest_ties_and_exact_tolerance(self):
        run = make_precision_probe_run(2**60)
        run["phase_reference"]["boundaries"][1]["timestamp_ns"] = 1024.5
        for policy, expected_round in (("earlier", "round-3"), ("later", "round-4")):
            with self.subTest(policy=policy):
                config = ReconstructionConfig(
                    max_alignment_error_ns=0.5, alignment_tie_policy=policy
                )
                segments = segment_probe_run(run, config)
                self.assertEqual(segments[1]["round_ids"][0], expected_round)
                self.assertEqual(segments[1]["alignment_error_ns"], 0.5)
                self.assertEqual(Fraction(segments[1]["alignment_error_ns"]), Fraction(1, 2))
        with self.assertRaisesRegex(ValidationError, "tie"):
            segment_probe_run(
                run, ReconstructionConfig(max_alignment_error_ns=0.5, alignment_tie_policy="reject")
            )
        with self.assertRaisesRegex(ValidationError, "alignment"):
            segment_probe_run(
                run, ReconstructionConfig(max_alignment_error_ns=math.nextafter(0.5, 0))
            )

    def test_large_epoch_collection_end_coverage_uses_exact_addition(self):
        run = make_precision_probe_run(2**60)
        run["phase_reference"]["collection_end_ns"] = 1028.25
        accepted = segment_probe_run(run, ReconstructionConfig(max_alignment_error_ns=0))
        self.assertEqual(accepted[-1]["round_ids"][-1], "round-7")
        run["phase_reference"]["collection_end_ns"] = 1028.0
        with self.assertRaisesRegex(ValidationError, "cover"):
            segment_probe_run(run, ReconstructionConfig(max_alignment_error_ns=0))
        # The former mixed subtraction/addition rounded this end past the last
        # observed round, even though the exact declared end equals that round.
        rounded_past_last = make_precision_probe_run(2**60)
        rounded_past_last["clock"]["probe_collection_start_ns"] += 200
        for record in rounded_past_last["probe_rounds"]:
            record["timestamp_ns"] += 200
        rounded_past_last["phase_reference"]["collection_end_ns"] = 1028.0
        with self.assertRaisesRegex(ValidationError, "cover"):
            segment_probe_run(rounded_past_last, ReconstructionConfig())

    def test_fractional_offsets_and_existing_ieee_values_roundtrip_exactly(self):
        for offset in (0.5, 0.25):
            with self.subTest(offset=offset):
                run = make_probe_run()
                run["clock"]["probe_collection_start_ns"] = offset
                run["clock"]["reference_collection_start_ns"] = 0.0
                for boundary in run["phase_reference"]["boundaries"]:
                    boundary["timestamp_ns"] += 100 - offset
                run["phase_reference"]["collection_end_ns"] += 100 - offset
                result = reconstruct_probe_run(run, ReconstructionConfig(max_alignment_error_ns=0))
                metadata = json.loads(json.dumps(result))["provenance"]
                self.assertEqual(Fraction(metadata["clock_alignment_offset_ns"]), Fraction(offset))
                self.assertTrue(all(segment["alignment_error_ns"] == 0 for segment in metadata["phase_segments"]))
                canonical_sha256(result)
        run = make_probe_run()
        run["clock"]["probe_collection_start_ns"] = 0.3
        run["clock"]["reference_collection_start_ns"] = 0.1
        for boundary in run["phase_reference"]["boundaries"]:
            boundary["timestamp_ns"] += 100
        run["phase_reference"]["collection_end_ns"] += 100
        exact_offset = Fraction(0.3) - Fraction(0.1)
        result = reconstruct_probe_run(run, ReconstructionConfig(max_alignment_error_ns=0.2))
        metadata = json.loads(json.dumps(result))["provenance"]
        self.assertEqual(Fraction(metadata["clock_alignment_offset_ns"]), exact_offset)
        self.assertNotEqual(Fraction(metadata["clock_alignment_offset_ns"]), Fraction("0.2"))
        self.assertTrue(all(
            Fraction(segment["alignment_error_ns"]) == exact_offset
            for segment in metadata["phase_segments"]
        ))

    def test_unrepresentable_fractional_offset_is_rejected_without_rounded_metadata(self):
        run = make_precision_probe_run(2**60)
        run["clock"]["reference_collection_start_ns"] = 0.5
        for operation in (segment_probe_run, reconstruct_probe_run):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(
                    ValidationError, "clock_alignment_offset_ns.*unsupported precision/range"
                ):
                    operation(run, ReconstructionConfig())
        overflow = make_probe_run()
        overflow["clock"]["probe_collection_start_ns"] = 1e308
        overflow["clock"]["reference_collection_start_ns"] = -1e308
        with self.assertRaisesRegex(ValidationError, "clock_alignment_offset_ns.*finite"):
            segment_probe_run(overflow, ReconstructionConfig())

    def test_unrepresentable_error_is_rejected_after_exact_tolerance_decision(self):
        run = make_precision_probe_run(2**60)
        run["clock"]["probe_collection_start_ns"] = 0
        run["phase_reference"]["boundaries"][0]["timestamp_ns"] = 0.5
        run["phase_reference"]["boundaries"][1]["timestamp_ns"] = 2**60 + 1024
        run["phase_reference"]["collection_end_ns"] = 2**60 + 2048
        with self.assertRaisesRegex(ValidationError, "max_alignment_error_ns"):
            segment_probe_run(
                run, ReconstructionConfig(max_alignment_error_ns=2**60 + 99)
            )
        for operation in (segment_probe_run, reconstruct_probe_run):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(
                    ValidationError, "alignment_error_ns.*unsupported precision/range"
                ):
                    operation(run, ReconstructionConfig(max_alignment_error_ns=2**60 + 100))

    def test_fractional_alignment_error_is_not_erased_by_large_offset(self):
        run = make_precision_probe_run(2**60)
        run["phase_reference"]["boundaries"][0]["timestamp_ns"] = 100.25
        with self.assertRaisesRegex(ValidationError, "alignment"):
            segment_probe_run(run, ReconstructionConfig(max_alignment_error_ns=0))
        accepted = reconstruct_probe_run(run, ReconstructionConfig(max_alignment_error_ns=0.25))
        self.assertEqual(accepted["provenance"]["phase_segments"][0]["alignment_error_ns"], 0.25)
        self.assertEqual(accepted["provenance"]["clock_alignment_offset_ns"], 2**60)
        encoded = json.loads(json.dumps(accepted))
        self.assertEqual(Fraction(encoded["provenance"]["phase_segments"][0]["alignment_error_ns"]), Fraction(1, 4))
        self.assertEqual(Fraction(encoded["provenance"]["clock_alignment_offset_ns"]), Fraction(2**60))
        canonical_sha256(accepted)

    def test_prefill_conserves_page_proxy_and_uses_explicit_long_tail(self):
        result = reconstruct_probe_run(make_probe_run(), ReconstructionConfig())
        profile = result["prefill_profile"][0][0]
        self.assertAlmostEqual(sum(profile[:4]), 12)
        self.assertAlmostEqual(sum(profile[4:8]), 6)
        self.assertEqual(profile[8:], [0.0] * 4)
        self.assertTrue(all(profile[index] > profile[index + 1] for index in range(3)))
        self.assertAlmostEqual(profile[0] / profile[3], 16)
        self.assertAlmostEqual(profile[4] / profile[7], 4)
        events = result["page_events"]["prefill"]
        self.assertEqual([event["relative_frequency_proxy"] for event in events[:3]], [12, 6, 0])

    def test_prefill_flat_exponent_and_zero_evictions(self):
        flat = reconstruct_probe_run(
            make_probe_run(), ReconstructionConfig(prefill_max_exponent=0)
        )
        self.assertEqual(flat["prefill_profile"][0][0][:4], [3.0] * 4)
        self.assertEqual(flat["prefill_profile"][0][0][4:8], [1.5] * 4)
        zero = make_probe_run()
        for round_record in zero["probe_rounds"]:
            for observation in round_record["observations"]:
                observation["evicted_probe_lines"] = 0
        profile = reconstruct_probe_run(zero, ReconstructionConfig())["prefill_profile"]
        self.assertTrue(all(value == 0 for layer in profile for head in layer for value in head))

    def test_decoding_noisy_votes_density_prefix_and_inactive_consistency(self):
        result = reconstruct_probe_run(make_probe_run(), ReconstructionConfig())
        pages = result["page_events"]["decoding"][0]["pages"]
        self.assertEqual([page["event"] for page in pages[:3]], [1, 1, 0])
        self.assertEqual(pages[0]["votes"], [1, 1, 0, 1, 1])
        self.assertEqual(pages[2]["votes"], [0, 1, 0, 0, 0])
        self.assertEqual([page["active_token_count"] for page in pages[:3]], [4, 3, 0])
        self.assertEqual(pages[0]["density"], 1)
        self.assertAlmostEqual(pages[1]["density"], 2 / 3)
        self.assertEqual(pages[2]["density"], 0.5)
        profile = result["decoding_steps"][0]["profile"][0]
        self.assertEqual(profile[0], [1.0] * 7 + [0.0] * 5)
        self.assertEqual(profile[1], [0.0] * 12)

    def test_density_is_local_within_head_and_radius_zero_is_explicit(self):
        run = make_probe_run()
        for round_record in run["probe_rounds"][3:]:
            for observation in round_record["observations"]:
                observation["reload_latency_ns"] = (
                    200 if observation["probe_id"] == "probe-0-1" else 100
                )
        local = reconstruct_probe_run(run, ReconstructionConfig())
        local_profile = local["decoding_steps"][0]["profile"][0]
        self.assertEqual(local_profile[0], [0.0] * 4 + [1.0] * 2 + [0.0] * 6)
        self.assertEqual(local_profile[1], [0.0] * 12)
        zero_radius = reconstruct_probe_run(run, ReconstructionConfig(decoding_density_radius=0))
        self.assertEqual(
            zero_radius["decoding_steps"][0]["profile"][0][0],
            [0.0] * 4 + [1.0] * 4 + [0.0] * 4,
        )

    def test_even_decode_vote_tie_is_rejected(self):
        run = make_probe_run()
        run["probe_rounds"].pop()
        for index, round_record in enumerate(run["probe_rounds"][3:]):
            for observation in round_record["observations"]:
                if observation["probe_id"] == "probe-0-0":
                    observation["reload_latency_ns"] = 200 if index < 2 else 100
        with self.assertRaisesRegex(ValidationError, "tie"):
            reconstruct_probe_run(run, ReconstructionConfig())

    def test_gold_changes_do_not_alter_features_and_inputs_are_not_mutated(self):
        run = make_probe_run()
        original = copy.deepcopy(run)
        baseline = reconstruct_probe_run(run, ReconstructionConfig())
        self.assertEqual(run, original)
        changed = copy.deepcopy(run)
        changed["gold"]["attributes"]["toy_topic"] = "beta"
        changed["gold"]["steps"][0]["gold_token_id"] = 17
        alternative = reconstruct_probe_run(changed, ReconstructionConfig())
        for field in ("prefill_profile", "decoding_steps", "page_events"):
            self.assertEqual(baseline[field], alternative[field])
        self.assertNotEqual(baseline["provenance"]["input_sha256"], alternative["provenance"]["input_sha256"])
        self.assertFalse(baseline["scientific_result"])
        self.assertEqual(baseline["phase_source_kind"], "oracle_aligned_phase")
        self.assertFalse(baseline["provenance"]["physical_addresses_recovered"])
        self.assertFalse(baseline["provenance"]["trace_only_phase_detection"])
        self.assertNotIn("gold", baseline)

    def test_declared_mapping_and_calibration_are_required_and_epoch_bound(self):
        missing = make_probe_run()
        missing["calibration"]["probes"].pop()
        with self.assertRaisesRegex(ValidationError, "bijection"):
            reconstruct_probe_run(missing, ReconstructionConfig())
        epoch = make_probe_run()
        epoch["layout"]["allocation_epoch"] = "other-epoch"
        with self.assertRaisesRegex(ValidationError, "allocation_epoch"):
            reconstruct_probe_run(epoch, ReconstructionConfig())

    def test_profile_axes_do_not_depend_on_page_or_probe_row_order(self):
        run = make_probe_run()
        baseline = reconstruct_probe_run(run, ReconstructionConfig())
        reordered = copy.deepcopy(run)
        reordered["layout"]["page_entries"].reverse()
        reordered["calibration"]["probes"].reverse()
        for round_record in reordered["probe_rounds"]:
            round_record["observations"].reverse()
        alternative = reconstruct_probe_run(reordered, ReconstructionConfig())
        self.assertEqual(baseline["prefill_profile"], alternative["prefill_profile"])
        self.assertEqual(baseline["decoding_steps"], alternative["decoding_steps"])
        self.assertTrue(
            all(math.isfinite(value) for layer in baseline["prefill_profile"] for head in layer for value in head)
        )


if __name__ == "__main__":
    unittest.main()
