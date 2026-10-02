"""CPU-only toy workload contracts; no model, physical probes or legacy execution."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from artifacts.janus_artifact.probe_contract import canonical_sha256, validate_probe_runs
from artifacts.janus_artifact.schema import ValidationError
from artifacts.janus_artifact.workload import (
    WorkloadConfig, generate_controlled_workload, integer_kv_aggregate,
    load_controlled_workload, toy_token_id_from_aggregate, toy_tokenizer_contract,
    validate_controlled_workload, write_controlled_workload,
)


class WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = generate_controlled_workload(WorkloadConfig())
        cls.small_config = WorkloadConfig(case_count=6, layers=2, kv_heads=2, prefill_rounds=3, noise_rate=0.0)
        cls.small = generate_controlled_workload(cls.small_config)

    def test_default_actual_groups_and_complete_probe_dimensions(self):
        self.assertEqual(Counter(run["split"] for run in self.default["runs"]),
                         {"train": 18, "validation": 3, "test": 3})
        self.assertEqual(len(self.default["runs"]), 24)
        self.assertTrue(all(len(run["probe_rounds"]) == 19 for run in self.default["runs"]))
        self.assertTrue(all(len(probe_round["observations"]) == 64 for run in self.default["runs"] for probe_round in run["probe_rounds"]))
        self.assertEqual(sum(len(probe_round["observations"]) for run in self.default["runs"] for probe_round in run["probe_rounds"]), 29184)
        self.assertEqual(self.default["split_contract"]["assignments_sha256"], "5183609f1ffa48b94ed350a5e9dab07026e9eb881b2fa7c5c62294e1ec8bdc01")
        self.assertEqual(len(validate_probe_runs(self.default["runs"])), 24)

    def test_integer_kv_gather_sum_and_symbolic_gold_interface(self):
        values = [-1, -1, 1, 1]
        self.assertEqual(integer_kv_aggregate([0, 1, 0], values), -3)
        self.assertEqual(integer_kv_aggregate([2, 3], values), 2)
        self.assertEqual(integer_kv_aggregate([0, 2], values), 0)
        self.assertEqual([toy_token_id_from_aggregate(value) for value in (-3, 2, 0)], [11, 17, 23])
        tokenizer = toy_tokenizer_contract()
        self.assertEqual(tokenizer["ordered_symbolic_table"], [
            {"token_id": 11, "symbol": "toy_alpha"}, {"token_id": 17, "symbol": "toy_beta"},
            {"token_id": 23, "symbol": "toy_zero_sum"}])
        tokenizer["ordered_symbolic_table"][0]["token_id"] = 99
        self.assertEqual(toy_tokenizer_contract()["ordered_symbolic_table"][0]["token_id"], 11)
        for accesses in (None, [], [-1], [4], [True]):
            with self.subTest(accesses=accesses), self.assertRaises(ValidationError):
                integer_kv_aggregate(accesses, values)
        with self.assertRaises(ValidationError):
            toy_token_id_from_aggregate(True)

    def test_ids_splits_and_gold_are_independent_of_noise(self):
        noisy = generate_controlled_workload(replace(self.small_config, noise_rate=1.0))
        self.assertEqual(noisy["split_contract"], self.small["split_contract"])
        self.assertEqual(noisy["tokenizer"], self.small["tokenizer"])
        for before, after in zip(self.small["runs"], noisy["runs"]):
            for key in ("case_id", "run_id", "sample_id", "response_id", "split", "split_assignment_id"):
                self.assertEqual(before[key], after[key])
            self.assertEqual(before["gold"], after["gold"])
            self.assertEqual(before["phase_reference"], after["phase_reference"])
            self.assertEqual(before["layout"], after["layout"])
            for round_before, round_after in zip(before["probe_rounds"], after["probe_rounds"]):
                self.assertEqual(round_before["round_id"], round_after["round_id"])
                for observation_before, observation_after in zip(round_before["observations"], round_after["observations"]):
                    self.assertEqual(observation_before["reload_latency_ns"] + observation_after["reload_latency_ns"], 300.0)

    def test_same_seed_deterministic_and_changed_seed_preserves_group_identity(self):
        self.assertEqual(self.small, generate_controlled_workload(self.small_config))
        other = generate_controlled_workload(replace(self.small_config, seed=20))
        self.assertEqual(self.small["split_contract"], other["split_contract"])
        self.assertEqual([run["sample_id"] for run in self.small["runs"]], [run["sample_id"] for run in other["runs"]])
        self.assertNotEqual(self.small["runs"][0]["probe_rounds"], other["runs"][0]["probe_rounds"])
        self.assertEqual([run["gold"] for run in self.small["runs"]], [run["gold"] for run in other["runs"]])

    def test_case_count_extension_does_not_reshuffle_existing_groups(self):
        extended = generate_controlled_workload(replace(self.small_config, case_count=8))
        self.assertEqual(self.small["split_contract"]["assignments"], extended["split_contract"]["assignments"][:6])
        self.assertEqual([run["case_id"] for run in self.small["runs"]], [run["case_id"] for run in extended["runs"][:6]])

    def test_frozen_candidate_ids_and_declared_test_only_oov(self):
        train_tokens = {step["gold_token_id"] for run in self.default["runs"] if run["split"] == "train" for step in run["gold"]["steps"]}
        self.assertEqual(train_tokens, {11, 17})
        for run in self.default["runs"]:
            self.assertEqual(run["gold"]["source_kind"], "oracle_annotation")
            if run["split"] == "test":
                self.assertEqual(run["gold"]["steps"][-1]["gold_token_id"], 23)
                self.assertNotIn(23, [step["gold_token_id"] for step in run["gold"]["steps"][:-1]])
            else:
                self.assertNotIn(23, [step["gold_token_id"] for step in run["gold"]["steps"]])
            self.assertEqual(run["gold"]["tokenizer"]["revision"], self.default["tokenizer"]["revision"])

    def test_raw_observables_and_oracle_annotations_are_separate(self):
        for run in self.small["runs"]:
            self.assertEqual(run["source_kind"], "synthetic_probe_simulation")
            self.assertIs(run["provenance"]["synthetic"], True)
            self.assertIs(run["provenance"]["scientific_result"], False)
            self.assertEqual(run["phase_reference"]["source_kind"], "oracle_annotation")
            self.assertEqual(run["gold"]["source_kind"], "oracle_annotation")
            self.assertIn("not_paper_physical_constants", run["provenance"]["simulation_parameters"]["parameter_status"])
            for probe_round in run["probe_rounds"]:
                self.assertEqual(set(probe_round), {"round_id", "timestamp_ns", "observations"})
                for observation in probe_round["observations"]:
                    self.assertEqual(set(observation), {"probe_id", "reload_latency_ns", "evicted_probe_lines"})
                    self.assertIn(observation["reload_latency_ns"], (100.0, 200.0))
                    self.assertGreaterEqual(observation["evicted_probe_lines"], 0)
                    self.assertLessEqual(observation["evicted_probe_lines"], 8)

    def test_real_toy_query_variation_produces_distinct_full_raw_profiles(self):
        # Full observed sequences differ through query-selected accesses, without an ID feature.
        profiles = []
        for run in self.default["runs"]:
            profiles.append(canonical_sha256([
                [{key: obs[key] for key in ("reload_latency_ns", "evicted_probe_lines")} for obs in probe_round["observations"]]
                for probe_round in run["probe_rounds"]
            ]))
        self.assertEqual(len(set(profiles)), 24)
        prefill = []
        for run in self.default["runs"]:
            prefill.append(canonical_sha256([
                [sum(run["probe_rounds"][round_index]["observations"][probe_index]["evicted_probe_lines"] for round_index in range(4))
                 for probe_index in range(64)]
            ]))
        self.assertEqual(len(set(prefill)), 24)

    def test_wrapper_rejects_changed_split_with_stale_or_rehashed_assignment_evidence(self):
        changed = deepcopy(self.small)
        original = changed["runs"][0]["split"]
        changed["runs"][0]["split"] = "train" if original != "train" else "test"
        with self.assertRaisesRegex(ValidationError, "frozen case assignment"):
            validate_controlled_workload(changed)
        changed["split_contract"]["assignments"][0]["split"] = changed["runs"][0]["split"]
        changed["split_contract"]["assignments_sha256"] = canonical_sha256(changed["split_contract"]["assignments"])
        previous = changed["split_contract"].pop("split_assignment_id")
        changed["split_contract"]["split_assignment_id"] = "toy-split-" + canonical_sha256(changed["split_contract"])
        self.assertNotEqual(previous, changed["split_contract"]["split_assignment_id"])
        with self.assertRaisesRegex(ValidationError, "fixed split contract"):
            validate_controlled_workload(changed)

    def test_wrapper_rejects_changed_config_tokenizer_or_geometry(self):
        for mutation in (
            lambda b: b["config"].update(noise_rate=0.5),
            lambda b: b["tokenizer"]["ordered_symbolic_table"][0].update(symbol="different"),
            lambda b: b["runs"][0]["calibration"]["probes"][0].update(translation_threshold_ns=160.0),
            lambda b: b["runs"][0]["gold"]["tokenizer"].update(revision="other"),
            lambda b: b["runs"][0].update(sample_id="new-id"),
        ):
            changed = deepcopy(self.small)
            mutation(changed)
            with self.assertRaises(ValidationError):
                validate_controlled_workload(changed)

    def test_wrapper_rejects_raw_only_tampering_even_inside_legal_numeric_ranges(self):
        changed = deepcopy(self.small)
        observation = changed["runs"][0]["probe_rounds"][0]["observations"][0]
        observation["reload_latency_ns"] = 300.0 - observation["reload_latency_ns"]
        observation["evicted_probe_lines"] = 8 - observation["evicted_probe_lines"]
        with self.assertRaisesRegex(ValidationError, "raw observations differ"):
            validate_controlled_workload(changed)

    def test_wrapper_rejects_raw_label_injection_and_physical_claim(self):
        changed = deepcopy(self.small)
        changed["runs"][0]["probe_rounds"][0]["observations"][0]["gold_token_id"] = 11
        with self.assertRaises(ValidationError):
            validate_controlled_workload(changed)
        changed = deepcopy(self.small)
        changed["runs"][0]["source_kind"] = "physical_probe_recording"
        with self.assertRaises(ValidationError):
            validate_controlled_workload(changed)

    def test_wrapper_rejects_oracle_and_timing_inconsistency(self):
        for mutation in (
            lambda b: b["runs"][0]["gold"]["steps"][0].update(gold_token_id=10),
            lambda b: b["runs"][0]["phase_reference"].update(reference_method="physical-detector"),
            lambda b: b["runs"][0]["probe_rounds"][0].update(round_id="other"),
            lambda b: b["runs"][0]["probe_rounds"][0]["observations"][0].update(reload_latency_ns=175.0),
        ):
            changed = deepcopy(self.small)
            mutation(changed)
            with self.assertRaises(ValidationError):
                validate_controlled_workload(changed)

    def test_legal_oracle_annotation_changes_preserve_all_raw_observables(self):
        changed = deepcopy(self.small)
        for run in changed["runs"]:
            run["gold"]["attributes"]["toy_topic"] = "beta" if run["gold"]["attributes"]["toy_topic"] == "alpha" else "alpha"
            for step in run["gold"]["steps"]:
                if step["gold_token_id"] != 23:
                    step["gold_token_id"] = 17 if step["gold_token_id"] == 11 else 11
        annotated = validate_controlled_workload(changed)
        self.assertNotEqual(canonical_sha256(self.small), canonical_sha256(annotated))
        self.assertEqual([run["probe_rounds"] for run in self.small["runs"]],
                         [run["probe_rounds"] for run in annotated["runs"]])
        self.assertEqual([run["phase_reference"] for run in self.small["runs"]],
                         [run["phase_reference"] for run in annotated["runs"]])
        self.assertEqual(annotated["runs"][0]["gold"], changed["runs"][0]["gold"])

    def test_validator_returns_canonical_copies_and_roundtrip_file(self):
        shuffled = deepcopy(self.small)
        shuffled["runs"].reverse()
        result = validate_controlled_workload(shuffled)
        self.assertEqual([run["case_id"] for run in result["runs"]], [row["case_id"] for row in result["split_contract"]["assignments"]])
        result["runs"][0]["gold"]["steps"][0]["gold_token_id"] = 17
        self.assertEqual(shuffled["runs"][-1]["gold"]["steps"][0]["gold_token_id"], 11)
        with tempfile.TemporaryDirectory() as directory:
            destination = write_controlled_workload(self.small, Path(directory) / "toy.json")
            self.assertEqual(load_controlled_workload(destination), self.small)

    def test_config_explicit_json_and_unknown_missing_duplicate_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(asdict(self.small_config)))
            self.assertEqual(WorkloadConfig.from_json(path), self.small_config)
            for mutation in (lambda c: c.pop("seed"), lambda c: c.update(extra=1)):
                value = asdict(self.small_config)
                mutation(value)
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValidationError, "explicitly contain all fields"):
                    WorkloadConfig.from_json(path)
            path.write_text('{"seed":19,"seed":20}')
            with self.assertRaisesRegex(ValidationError, "duplicate workload JSON key"):
                WorkloadConfig.from_json(path)
            path.write_text('{"noise_rate":NaN}')
            with self.assertRaisesRegex(ValidationError, "nonfinite"):
                WorkloadConfig.from_json(path)

    def test_invalid_config_and_budgets_rejected_before_generation(self):
        for name, value in (("case_count", 0), ("steps_per_response", True), ("seed", 1 << 80),
                            ("split_seed", ""), ("noise_rate", float("nan")), ("noise_rate", float("inf")),
                            ("noise_rate", 10 ** 1000), ("noise_rate", -0.1), ("noise_rate", 1.1),
                            ("pages_per_head", 1), ("layers", 1000000)):
            config = replace(self.small_config, **{name: value})
            if name == "pages_per_head":
                config = replace(config, tokens_per_page=1)
            with self.subTest(name=name, value=value), self.assertRaises(ValidationError):
                config.validate()
        with patch("artifacts.janus_artifact.workload.MAX_WORKLOAD_OBSERVATIONS", 1):
            with self.assertRaisesRegex(ValidationError, "observation budget"):
                generate_controlled_workload(self.small_config)

    def test_oversized_generation_fails_before_run_or_observation_allocation(self):
        with patch("artifacts.janus_artifact.workload._generate_run") as generate_run:
            with self.assertRaisesRegex(ValidationError, "serialized byte budget"):
                generate_controlled_workload(WorkloadConfig(case_count=800))
            generate_run.assert_not_called()

    def test_file_budget_duplicate_keys_and_wrapper_extra_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text(json.dumps(self.small))
            with patch("artifacts.janus_artifact.workload.MAX_WORKLOAD_FILE_BYTES", 16):
                with self.assertRaisesRegex(ValidationError, "byte budget"):
                    load_controlled_workload(path)
            path.write_text('{"schema_version":"x","schema_version":"y"}')
            with self.assertRaisesRegex(ValidationError, "duplicate workload JSON key"):
                load_controlled_workload(path)
        changed = deepcopy(self.small)
        changed["extra"] = True
        with self.assertRaisesRegex(ValidationError, "fields must equal"):
            validate_controlled_workload(changed)


if __name__ == "__main__":
    unittest.main()
