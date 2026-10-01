import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from artifacts.janus_artifact.atr_data import load_atr_records
from artifacts.janus_artifact.probe_reconstruction import ReconstructionConfig
from artifacts.janus_artifact.qai import load_qai_records
from artifacts.janus_artifact.schema import ValidationError, load_jsonl, validate_trace_records
from artifacts.janus_artifact.upstream_bridge import (
    QAI_QUANTIZATION_MAX,
    _qai_rank_surrogate,
    _trace_records,
    build_controlled_artifacts,
    run_upstream_smoke,
)
from artifacts.janus_artifact.workload import WorkloadConfig, generate_controlled_workload


class ControlledBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture.name)
        cls.bundle = generate_controlled_workload(WorkloadConfig())
        cls.config = ReconstructionConfig()
        cls.build = build_controlled_artifacts(cls.bundle, cls.config, cls.root / "artifacts")

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.destination = Path(self.work.name) / "output"

    def test_build_preserves_every_identity_and_preassigned_split(self):
        self.assertEqual(self.build["split_response_counts"], {"train": 18, "validation": 3, "test": 3})
        self.assertEqual(self.build["probe_runs"], 24)
        self.assertEqual(self.build["qai_records"], 24)
        self.assertEqual(self.build["atr_responses"], 24)
        self.assertEqual(self.build["atr_steps"], 72)
        self.assertEqual(self.build["candidate_token_ids_train_only"], [11, 17])
        source = {run["response_id"]: run for run in self.bundle["runs"]}
        atr = load_atr_records(self.build["paths"]["atr_manifest"])
        for record in atr:
            original = source[record["response_id"]]
            self.assertEqual((record["case_id"], record["split"]), (original["case_id"], original["split"]))
            self.assertEqual(record["tokenizer"], original["gold"]["tokenizer"])
            self.assertEqual(record["alignment"], original["gold"]["alignment"])
            self.assertEqual(record["provenance"]["split_assignment_id"], original["split_assignment_id"])
            for actual, gold in zip(record["steps"], original["gold"]["steps"]):
                self.assertEqual({key: actual[key] for key in ("step_id", "step_index", "gold_token_id")}, gold)
        source_samples = {run["sample_id"]: run for run in self.bundle["runs"]}
        for record in load_qai_records(self.build["paths"]["qai_manifest"]):
            original = source_samples[record.sample_id]
            self.assertEqual((record.case_id, record.split, record.label), (original["case_id"], original["split"], original["gold"]["attributes"]["toy_topic"]))

    def test_page_and_token_replays_are_explicitly_reconstructed(self):
        traces = load_jsonl(self.build["paths"]["traces"])
        replayed = load_jsonl(self.build["paths"]["replayed_traces"])
        validate_trace_records(traces)
        validate_trace_records(replayed)
        self.assertEqual(len(traces), 96)
        self.assertEqual(len(replayed), 96)
        self.assertEqual({(row["phase"], row["trace"]["granularity"]) for row in traces}, {
            ("prefill", "page"), ("prefill", "token"), ("decoding", "page"), ("decoding", "token"),
        })
        by_id = {row["trace_id"]: row for row in traces}
        for row in replayed:
            self.assertEqual(row["source_kind"], "replay_derived")
            self.assertFalse(row["provenance"]["real_side_channel_reproduction"])
            parent = by_id[row["provenance"]["parent_trace_ids"][0]]
            self.assertEqual(row["trace"], parent["trace"])
            self.assertEqual(row["trace"]["phase_source_kind"], "oracle_aligned_phase")
            self.assertEqual(row["trace"]["data_stage"], f"reconstructed_{row['trace']['granularity']}_sparsity")
            self.assertEqual(parent["labels"]["source_kind"], "oracle_annotation")

    def test_page_mapping_order_is_bound_to_each_numeric_position(self):
        for row in load_jsonl(self.build["paths"]["traces"]):
            trace = row["trace"]
            if trace["granularity"] != "page":
                continue
            self.assertEqual(len(trace["ordered_page_ids"]), 64)
            self.assertEqual(len(set(trace["ordered_page_ids"])), 64)
            vectors = [trace["observations"]] if row["phase"] == "prefill" else trace["observations"]
            self.assertTrue(all(len(vector) == len(trace["ordered_page_ids"]) for vector in vectors))
        reconstruction = deepcopy(load_jsonl(self.build["paths"]["reconstructed_profiles"])[0])
        reconstruction["page_events"]["decoding"][0]["pages"].reverse()
        with self.assertRaisesRegex(ValidationError, "observation order changes"):
            _trace_records(self.bundle["runs"][0], reconstruction)

    def test_qai_payloads_are_known_numeric_surrogates_without_identity_arrays(self):
        for row in load_jsonl(self.build["paths"]["qai_manifest"]):
            path = Path(self.build["paths"]["qai_manifest"]).parent / row["npz_path"]
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(set(payload.files), {"attn_rank", "top_k"})
                self.assertEqual(payload["attn_rank"].shape, (4, 4, 1, 8))
                self.assertEqual(payload["attn_rank"].dtype, np.dtype("uint16"))
                self.assertEqual(payload["top_k"].item(), 2)
        adapter = self.build["qai_adapter"]
        self.assertEqual(adapter["quantization"]["maximum"], QAI_QUANTIZATION_MAX)
        self.assertEqual(adapter["qai_selected_ranks"], 2)
        self.assertFalse(adapter["original_attention_ranks"])
        self.assertTrue(adapter["information_loss"])

    def test_only_oracle_labels_change_when_gold_annotations_change(self):
        changed = deepcopy(self.bundle)
        for run in changed["runs"]:
            attributes = run["gold"]["attributes"]
            attributes["toy_topic"] = "beta" if attributes["toy_topic"] == "alpha" else "alpha"
            for step in run["gold"]["steps"]:
                if step["gold_token_id"] in (11, 17):
                    step["gold_token_id"] = 17 if step["gold_token_id"] == 11 else 11
        actual = build_controlled_artifacts(changed, self.config, self.destination)
        self.assertNotEqual(actual["input_bundle_sha256"], self.build["input_bundle_sha256"])
        baseline_traces = {row["trace_id"]: row for row in load_jsonl(self.build["paths"]["traces"])}
        for row in load_jsonl(actual["paths"]["traces"]):
            self.assertEqual(row["trace"], baseline_traces[row["trace_id"]]["trace"])
            self.assertNotEqual(row["labels"], baseline_traces[row["trace_id"]]["labels"])
        baseline_atr = {row["response_id"]: row for row in load_atr_records(self.build["paths"]["atr_manifest"])}
        for row in load_atr_records(actual["paths"]["atr_manifest"]):
            original = baseline_atr[row["response_id"]]
            self.assertEqual([step["profile"] for step in row["steps"]], [step["profile"] for step in original["steps"]])
        baseline_qai = {row["sample_id"]: row for row in load_jsonl(self.build["paths"]["qai_manifest"])}
        for row in load_jsonl(actual["paths"]["qai_manifest"]):
            old = Path(self.build["paths"]["qai_manifest"]).parent / baseline_qai[row["sample_id"]]["npz_path"]
            new = Path(actual["paths"]["qai_manifest"]).parent / row["npz_path"]
            with np.load(old, allow_pickle=False) as before, np.load(new, allow_pickle=False) as after:
                np.testing.assert_array_equal(before["attn_rank"], after["attn_rank"])
                self.assertEqual(before["top_k"].item(), after["top_k"].item())

    def test_unknown_wrapper_field_rejected_before_output_or_reconstruction(self):
        bundle = deepcopy(self.bundle)
        bundle["unknown"] = True
        with patch("artifacts.janus_artifact.upstream_bridge.reconstruct_probe_run", side_effect=AssertionError("must not reconstruct")):
            with self.assertRaises(ValidationError):
                build_controlled_artifacts(bundle, self.config, self.destination)
        self.assertFalse(self.destination.exists())

    def test_physical_source_cannot_be_masqueraded_as_synthetic_training(self):
        bundle = deepcopy(self.bundle)
        run = bundle["runs"][0]
        run["source_kind"] = "physical_probe_recording"
        run["provenance"] = {"collection_run_id": "declared-test-run", "collector_revision": "declared-test-v1"}
        run["layout"]["source_kind"] = "calibrated_logical_mapping"
        run["calibration"]["source_kind"] = "physical_contention_calibration"
        with self.assertRaises(ValidationError):
            build_controlled_artifacts(bundle, self.config, self.destination)
        self.assertFalse(self.destination.exists())

    def test_all_oracle_qai_labels_one_class_rejected_before_any_output(self):
        bundle = deepcopy(self.bundle)
        for run in bundle["runs"]:
            run["gold"]["attributes"]["toy_topic"] = "alpha"
        with self.assertRaisesRegex(ValidationError, "at least two train labels"):
            build_controlled_artifacts(bundle, self.config, self.destination)
        self.assertFalse(self.destination.exists())

    def test_nontrain_class_cannot_compensate_for_single_train_class(self):
        bundle = deepcopy(self.bundle)
        for run in bundle["runs"]:
            run["gold"]["attributes"]["toy_topic"] = "alpha" if run["split"] == "train" else "beta"
        with self.assertRaisesRegex(ValidationError, "at least two train labels"):
            build_controlled_artifacts(bundle, self.config, self.destination)
        self.assertFalse(self.destination.exists())

    def test_existing_output_and_files_are_preserved(self):
        self.destination.mkdir()
        marker = self.destination / "user-file"
        marker.write_text("preserve me")
        with self.assertRaisesRegex(ValidationError, "must be new"):
            build_controlled_artifacts(self.bundle, self.config, self.destination)
        self.assertEqual(marker.read_text(), "preserve me")

    def test_duplicate_numeric_rank_payloads_do_not_use_ids_to_bypass_guard(self):
        real_adapter = _qai_rank_surrogate
        def zero_adapter(profile):
            ranks, metadata = real_adapter(profile)
            return np.zeros_like(ranks), metadata
        with patch("artifacts.janus_artifact.upstream_bridge._qai_rank_surrogate", zero_adapter):
            with self.assertRaisesRegex(ValidationError, "payload split leakage"):
                build_controlled_artifacts(self.bundle, self.config, self.destination)
        self.assertFalse(self.destination.exists())

    def test_quantization_is_per_run_and_handles_zero(self):
        ranks, metadata = _qai_rank_surrogate([[[0.0, 1.0, 2.0, 3.0]]])
        np.testing.assert_array_equal(ranks[0, 0, 0], [0, 21845, 43690, 65535])
        self.assertEqual(metadata["input_maximum"], 3.0)
        zero, metadata = _qai_rank_surrogate([[[0.0, 0.0]]])
        self.assertFalse(np.any(zero))
        self.assertEqual(metadata["input_maximum"], 0.0)

    def test_adapter_rejects_nonfinite_negative_and_non_3d_profiles(self):
        for profile in ([[[float("nan")]]], [[[float("inf")]]], [[[-1.0]]], [[1.0, 2.0]]):
            with self.subTest(profile=profile), self.assertRaises(ValidationError):
                _qai_rank_surrogate(profile)

    def test_config_and_cpu_scope_are_explicit(self):
        with self.assertRaisesRegex(ValidationError, "ReconstructionConfig"):
            build_controlled_artifacts(self.bundle, {}, self.destination)
        with self.assertRaisesRegex(ValidationError, "CPU"):
            run_upstream_smoke(self.destination, device="cuda")
        self.assertFalse(self.destination.exists())

    def test_actual_small_training_reload_and_full_denominators(self):
        original_load = torch.load
        with patch("artifacts.janus_artifact.upstream_bridge.torch.load", wraps=original_load) as safe_load:
            report = run_upstream_smoke(self.destination)
        self.assertGreaterEqual(safe_load.call_count, 4)
        self.assertTrue(all(call.kwargs.get("weights_only") is True for call in safe_load.call_args_list))
        for name in ("qai_training", "atr_training"):
            self.assertTrue(report[name]["gradient_observed"])
            self.assertTrue(report[name]["parameter_changed"])
            self.assertEqual(len(report[name]["history"]), 1)
        self.assertTrue(report["qai_checkpoint_reload_verified"])
        self.assertTrue(report["atr_checkpoint_reload_verified"])
        self.assertEqual(report["pasr_smoke_only"]["micro_denominator_present_queries"], 3)
        self.assertEqual(report["dasr_smoke_only"]["micro_denominator_all_gold_tokens"], 9)
        self.assertEqual(report["dasr_smoke_only"]["out_of_vocabulary_gold_tokens_counted_incorrect"], 3)
        self.assertEqual(report["atr_test_predictions"], 9)
        self.assertFalse(report["scientific_result"])
        self.assertFalse(report["paper_reproduction"])
        saved = json.loads((self.destination / "report.json").read_text())
        self.assertEqual(saved["status"], "ok")


if __name__ == "__main__":
    unittest.main()
