"""CPU-only synthetic and offline-contract regression tests."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from artifacts.janus_artifact import probe_contract as contract
from artifacts.janus_artifact.schema import ValidationError


def valid_run(prefix="case-α", split="train"):
    probes = ["probe-a", "probe-b"]
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "run_id": f"{prefix}-run",
        "source_kind": "synthetic_probe_simulation",
        "provenance": {"synthetic": True, "simulator": "CPU unit fixture"},
        "task_id": "toy-task",
        "case_id": prefix,
        "sample_id": f"{prefix}-sample",
        "response_id": f"{prefix}-response",
        "split": split,
        "split_assignment_id": "fixed-assignment",
        "allocation_epoch": f"{prefix}-epoch",
        "clock": {
            "unit": "ns",
            "probe_collection_start_ns": 1000,
            "reference_collection_start_ns": 2000,
        },
        "layout": {
            "source_kind": "synthetic_layout",
            "calibration_id": f"{prefix}-calibration",
            "allocation_epoch": f"{prefix}-epoch",
            "layer_ids": ["layer-0"],
            "kv_head_ids": ["head-0"],
            "token_width": 4,
            "page_entries": [
                {
                    "page_id": f"page-{index}", "probe_id": probe,
                    "layer_id": "layer-0", "kv_head_id": "head-0",
                    "page_order": index, "token_start": index * 2, "token_count": 2,
                }
                for index, probe in enumerate(probes)
            ],
        },
        "calibration": {
            "source_kind": "synthetic_calibration",
            "calibration_id": f"{prefix}-calibration",
            "allocation_epoch": f"{prefix}-epoch",
            "probes": [
                {"probe_id": probe, "translation_threshold_ns": 150, "probe_set_line_count": 8}
                for probe in probes
            ],
        },
        "probe_rounds": [
            {
                "round_id": f"{prefix}-round-{index}", "timestamp_ns": 1000 + index * 10,
                "observations": [
                    {"probe_id": probe, "reload_latency_ns": 100 + 100 * (index % 2),
                     "evicted_probe_lines": index % 3}
                    for probe in probes
                ],
            }
            for index in range(5)
        ],
        "phase_reference": {
            "source_kind": "oracle_annotation",
            "reference_method": "synthetic schedule",
            "alignment_evidence_id": f"{prefix}-alignment",
            "boundaries": [
                {"timestamp_ns": 2000, "phase": "prefill", "step_id": None, "step_index": None},
                {"timestamp_ns": 2020, "phase": "decoding", "step_id": f"{prefix}-step-0", "step_index": 0},
                {"timestamp_ns": 2040, "phase": "decoding", "step_id": f"{prefix}-step-1", "step_index": 1},
            ],
            "collection_end_ns": 2060,
        },
        "gold": {
            "source_kind": "oracle_annotation",
            "oracle_method": "synthetic exact integer schedule",
            "attributes": {"toy_topic": "alpha"},
            "tokenizer": {"name": "toy-symbols", "revision": "unit-v1", "vocab_size": 24},
            "alignment": {
                "step_index_base": 0, "profile_predicts": "same_index_output_token",
                "bos_included": False, "eos_included": False,
                "special_tokens": "excluded", "response_scope": "complete_response",
            },
            "steps": [
                {"step_id": f"{prefix}-step-0", "step_index": 0, "gold_token_id": 11},
                {"step_id": f"{prefix}-step-1", "step_index": 1, "gold_token_id": 17},
            ],
        },
    }


class ProbeContractTests(unittest.TestCase):
    def assert_invalid(self, run, message):
        with mock.patch.object(contract, "deepcopy", side_effect=AssertionError("copied invalid data")):
            with self.assertRaisesRegex(ValidationError, message):
                contract.validate_probe_run(run)

    def test_valid_run_is_independent_and_unicode_is_preserved(self):
        original = valid_run()
        checked = contract.validate_probe_run(original)
        self.assertEqual(checked, original)
        self.assertIsNot(checked, original)
        checked["gold"]["steps"][0]["gold_token_id"] = 12
        self.assertEqual(original["gold"]["steps"][0]["gold_token_id"], 11)
        self.assertEqual(checked["case_id"], "case-α")

    def test_source_classes_and_physical_evidence_are_required(self):
        for change, message in (
            (lambda run: run.update(source_kind="oracle_annotation"), "raw source_kind"),
            (lambda run: run["provenance"].update(synthetic=False), "synthetic=true"),
            (lambda run: run["layout"].update(source_kind="calibrated_logical_mapping"), "incompatible"),
            (lambda run: run["calibration"].update(source_kind="physical_contention_calibration"), "incompatible"),
        ):
            with self.subTest(message=message):
                run = valid_run()
                change(run)
                self.assert_invalid(run, message)
        physical = valid_run()
        physical["source_kind"] = "physical_probe_recording"
        physical["layout"]["source_kind"] = "calibrated_logical_mapping"
        physical["calibration"]["source_kind"] = "physical_contention_calibration"
        physical["provenance"] = {"collection_run_id": "declared-collection", "collector_revision": "revision-1"}
        self.assertEqual(contract.validate_probe_run(physical), physical)
        for missing in ("collector_revision", "collection_run_id"):
            with self.subTest(missing=missing):
                run = deepcopy(physical)
                del run["provenance"][missing]
                self.assert_invalid(run, missing)
        physical["provenance"]["synthetic"] = True
        self.assert_invalid(physical, "physical probes cannot claim synthetic")

    def test_raw_fields_cannot_embed_oracle_phase_gold_or_labels(self):
        for target, field in (("run", "labels"), ("round", "phase"), ("observation", "gold_token_id")):
            with self.subTest(target=target):
                run = valid_run()
                obj = run if target == "run" else run["probe_rounds"][0]
                if target == "observation":
                    obj = obj["observations"][0]
                obj[field] = "oracle"
                self.assert_invalid(run, "fields must equal")

    def test_physical_missing_mapping_calibration_or_epoch_rejected(self):
        for section, field in (("run", "layout"), ("run", "calibration"), ("run", "allocation_epoch"),
                               ("layout", "allocation_epoch"), ("calibration", "calibration_id")):
            with self.subTest(section=section, field=field):
                run = valid_run()
                obj = run if section == "run" else run[section]
                del obj[field]
                self.assert_invalid(run, "fields must equal")

    def test_allocation_epoch_and_calibration_id_cannot_be_reused_mismatched(self):
        for section, field in (("layout", "allocation_epoch"), ("calibration", "allocation_epoch"),
                               ("layout", "calibration_id")):
            with self.subTest(section=section, field=field):
                run = valid_run()
                run[section][field] = "another-allocation"
                self.assert_invalid(run, "mismatch")

    def test_mapping_axes_ids_and_every_head_are_complete(self):
        cases = []
        run = valid_run()
        run["layout"]["layer_ids"].append("layer-0")
        cases.append((run, "duplicate IDs"))
        run = valid_run()
        run["layout"]["kv_head_ids"].append("head-1")
        cases.append((run, "every declared layer/head"))
        run = valid_run()
        run["layout"]["page_entries"][1]["page_id"] = "page-0"
        cases.append((run, "duplicate page_id"))
        run = valid_run()
        run["layout"]["page_entries"][1]["probe_id"] = "probe-a"
        cases.append((run, "duplicate page_id or probe_id"))
        for run, message in cases:
            with self.subTest(message=message):
                self.assert_invalid(run, message)

    def test_logical_page_order_and_token_intervals_are_exact(self):
        for field, value, message in (
            ("page_order", 0, "contiguous"), ("token_start", 1, "contiguous"),
            ("token_count", 1, "exactly cover"), ("token_count", True, "integer"),
        ):
            with self.subTest(field=field, value=value):
                run = valid_run()
                run["layout"]["page_entries"][1][field] = value
                self.assert_invalid(run, message)

    def test_calibration_probes_form_bijection(self):
        for mutation in ("duplicate", "unknown", "missing"):
            with self.subTest(mutation=mutation):
                run = valid_run()
                probes = run["calibration"]["probes"]
                if mutation == "duplicate":
                    probes[1]["probe_id"] = probes[0]["probe_id"]
                elif mutation == "unknown":
                    probes[1]["probe_id"] = "unmapped"
                else:
                    probes.pop()
                self.assert_invalid(run, "duplicate|bijection")

    def test_every_round_has_unique_complete_known_probes(self):
        for mutation in ("duplicate", "unknown", "missing"):
            with self.subTest(mutation=mutation):
                run = valid_run()
                observations = run["probe_rounds"][0]["observations"]
                if mutation == "duplicate":
                    observations[1]["probe_id"] = observations[0]["probe_id"]
                elif mutation == "unknown":
                    observations[1]["probe_id"] = "unmapped"
                else:
                    observations.pop()
                self.assert_invalid(run, "duplicate|unknown|all calibrated")

    def test_observed_evictions_are_bounded_by_the_calibrated_probe_set(self):
        for value in (-1, 9, True, 1.5):
            with self.subTest(value=value):
                run = valid_run()
                run["probe_rounds"][0]["observations"][0]["evicted_probe_lines"] = value
                self.assert_invalid(run, "evicted_probe_lines.*integer")

    def test_finite_ns_values_reject_bool_nan_inf_and_negative_latency(self):
        for value in (True, float("nan"), float("inf"), -1):
            with self.subTest(value=value):
                run = valid_run()
                run["probe_rounds"][0]["observations"][0]["reload_latency_ns"] = value
                self.assert_invalid(run, "finite|nonnegative")
        run = valid_run()
        run["clock"]["unit"] = "seconds"
        self.assert_invalid(run, "clock.unit")
        run = valid_run()
        run["calibration"]["probes"][0]["translation_threshold_ns"] = -1
        self.assert_invalid(run, "nonnegative")

    def test_round_ids_and_times_are_strict(self):
        run = valid_run()
        run["probe_rounds"][1]["round_id"] = run["probe_rounds"][0]["round_id"]
        self.assert_invalid(run, "duplicate round_id")
        for value in (1000, 999):
            with self.subTest(value=value):
                run = valid_run()
                run["probe_rounds"][1]["timestamp_ns"] = value
                self.assert_invalid(run, "strictly ascending")

    def test_reference_and_gold_are_explicit_oracle_annotations(self):
        for section in ("phase_reference", "gold"):
            with self.subTest(section=section):
                run = valid_run()
                run[section]["source_kind"] = "physical_probe_recording"
                self.assert_invalid(run, "oracle_annotation")

    def test_reference_phase_order_and_step_sequence_are_strict(self):
        run = valid_run()
        run["phase_reference"]["boundaries"][0]["step_index"] = 0
        self.assert_invalid(run, "prefill with null")
        run = valid_run()
        run["phase_reference"]["boundaries"][2]["phase"] = "prefill"
        self.assert_invalid(run, "must be decoding")
        run = valid_run()
        run["phase_reference"]["boundaries"][2]["step_index"] = 5
        self.assert_invalid(run, "contiguous from zero")
        run = valid_run()
        run["phase_reference"]["boundaries"][2]["timestamp_ns"] = 2020
        self.assert_invalid(run, "strictly ascending")
        run = valid_run()
        run["phase_reference"]["collection_end_ns"] = 2040
        self.assert_invalid(run, "follow the last boundary")

    def test_gold_matches_complete_reference_steps_and_declared_tokenizer(self):
        for mutation, message in (
            ("missing", "exactly the same"), ("identity", "identity or order"),
            ("outside_vocab", "gold_token_id.*integer"), ("boolean_index", "step_index.*integer"),
            ("alignment", "alignment"), ("unknown_attribute", "toy_topic"),
        ):
            with self.subTest(mutation=mutation):
                run = valid_run()
                gold = run["gold"]
                if mutation == "missing":
                    gold["steps"].pop()
                elif mutation == "identity":
                    gold["steps"][0]["step_id"] = "another-step"
                elif mutation == "outside_vocab":
                    gold["steps"][0]["gold_token_id"] = 24
                elif mutation == "boolean_index":
                    gold["steps"][0]["step_index"] = False
                elif mutation == "alignment":
                    gold["alignment"]["bos_included"] = True
                else:
                    gold["attributes"]["toy_topic"] = "gamma"
                self.assert_invalid(run, message)

    def test_json_primitive_cycles_classes_depth_and_integer_budgets(self):
        for value, message in (
            ({"tuple": (1, 2)}, "builtin primitive"),
            ({"huge": 2**100}, "64-bit"),
            ({"invalid_unicode": "\ud800"}, "Unicode"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValidationError, message):
                    contract.canonical_sha256(value)
        cycle = []
        cycle.append(cycle)
        with self.assertRaisesRegex(ValidationError, "cycles"):
            contract.canonical_sha256(cycle)
        with mock.patch.object(contract, "MAX_JSON_DEPTH", 2):
            self.assert_invalid(valid_run(), "depth budget")
        with mock.patch.object(contract, "MAX_PRIMITIVE_NODES", 10):
            self.assert_invalid(valid_run(), "node/depth budget")
        with mock.patch.object(contract, "MAX_STRING_BYTES", 2):
            self.assert_invalid(valid_run(), "string budget")

    def test_observation_and_reconstructed_tensor_budgets_precede_copy(self):
        with mock.patch.object(contract, "MAX_OBSERVATIONS", 8):
            self.assert_invalid(valid_run(), "observation budget")
        with mock.patch.object(contract, "MAX_RECONSTRUCTED_CELLS", 8):
            self.assert_invalid(valid_run(), "reconstructed tensor budget")
        runs = [valid_run("a"), valid_run("b")]
        with mock.patch.object(contract, "MAX_OBSERVATIONS", 12):
            with self.assertRaisesRegex(ValidationError, "total observation budget"):
                contract.validate_probe_runs(runs)
        with mock.patch.object(contract, "MAX_RECONSTRUCTED_CELLS", 20):
            with self.assertRaisesRegex(ValidationError, "total reconstructed tensor budget"):
                contract.validate_probe_runs(runs)

    def test_global_run_sample_response_and_step_ids_are_unique(self):
        for field in ("run_id", "sample_id", "response_id", "step_id"):
            with self.subTest(field=field):
                first, second = valid_run("a"), valid_run("b")
                if field == "step_id":
                    step_id = first["gold"]["steps"][0]["step_id"]
                    second["gold"]["steps"][0]["step_id"] = step_id
                    second["phase_reference"]["boundaries"][1]["step_id"] = step_id
                else:
                    second[field] = first[field]
                with self.assertRaisesRegex(ValidationError, f"duplicate {field}"):
                    contract.validate_probe_runs([first, second])

    def test_case_split_isolation_allows_multiple_same_split_samples(self):
        first, second = valid_run("a"), valid_run("b")
        second["case_id"] = first["case_id"]
        checked = contract.validate_probe_runs([first, second])
        self.assertEqual(len(checked), 2)
        second["split"] = "test"
        with self.assertRaisesRegex(ValidationError, "case split leakage"):
            contract.validate_probe_runs([first, second])

    def test_canonical_hash_is_order_independent_but_preserves_array_order(self):
        self.assertEqual(len(contract.canonical_sha256({"seed": -(2**63)})), 64)
        self.assertEqual(
            contract.canonical_sha256({"β": 2, "α": 1}),
            contract.canonical_sha256({"α": 1, "β": 2}),
        )
        self.assertNotEqual(contract.canonical_sha256([1, 2]), contract.canonical_sha256([2, 1]))


class ProbeLoadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "probe.json"

    def generated_wrapper(self):
        from artifacts.janus_artifact.workload import WorkloadConfig, generate_controlled_workload

        return generate_controlled_workload(WorkloadConfig(
            case_count=2, steps_per_response=1, layers=1, kv_heads=1,
            pages_per_head=2, tokens_per_page=1, prefill_rounds=1,
            decode_rounds=3, noise_rate=0,
        ))

    def test_load_single_array_jsonl_and_validated_workload_extract(self):
        runs = [valid_run("a"), valid_run("b")]
        wrapper = self.generated_wrapper()
        cases = (
            (json.dumps(runs[0], ensure_ascii=False), [runs[0]]),
            (json.dumps(runs, ensure_ascii=False), runs),
            ("\n\n".join(json.dumps(run, ensure_ascii=False) for run in runs) + "\n", runs),
            (json.dumps(wrapper, ensure_ascii=False), wrapper["runs"]),
        )
        for text, expected in cases:
            with self.subTest(text=text[:20]):
                self.path.write_text(text, encoding="utf-8")
                self.assertEqual(contract.load_probe_runs(self.path), expected)

    def test_loader_rejects_duplicate_keys_nonfinite_empty_and_malformed(self):
        for text, message in (
            ('{"run_id":"a","run_id":"b"}', "duplicate object fields"),
            ('{"value":NaN}', "non-finite"),
            ('{"value":1e1000}', "finite"),
            (" \n ", "non-empty"),
            ('{"run_id":', "neither valid JSON nor JSONL"),
        ):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValidationError, message):
                    contract.load_probe_runs(self.path)

    def test_loader_checks_file_budget_before_read(self):
        self.path.write_text(" " * 101, encoding="utf-8")
        with mock.patch.object(contract, "MAX_FILE_BYTES", 100):
            with mock.patch.object(Path, "open", side_effect=AssertionError("read oversized file")):
                with self.assertRaisesRegex(ValidationError, "file-size budget"):
                    contract.load_probe_runs(self.path)

    def test_wrapper_does_not_make_unknown_raw_fields_valid(self):
        wrapper = self.generated_wrapper()
        wrapper["runs"][0]["physical_address"] = "not part of logical mapping contract"
        self.path.write_text(json.dumps(wrapper), encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "probe run fields"):
            contract.load_probe_runs(self.path)


    def test_workload_wrapper_cannot_bypass_its_frozen_split_proof(self):
        for mutation, message in (
            ("run_split", "frozen case assignment"),
            ("assignment", "fixed split contract/digest"),
            ("placeholder", "config"),
        ):
            with self.subTest(mutation=mutation):
                wrapper = self.generated_wrapper()
                if mutation == "run_split":
                    old_split = wrapper["runs"][0]["split"]
                    wrapper["runs"][0]["split"] = "test" if old_split != "test" else "train"
                elif mutation == "assignment":
                    old_split = wrapper["split_contract"]["assignments"][0]["split"]
                    wrapper["split_contract"]["assignments"][0]["split"] = "test" if old_split != "test" else "train"
                else:
                    wrapper["config"] = {}
                self.path.write_text(json.dumps(wrapper), encoding="utf-8")
                with self.assertRaisesRegex(ValidationError, message):
                    contract.load_probe_runs(self.path)


if __name__ == "__main__":
    unittest.main()
