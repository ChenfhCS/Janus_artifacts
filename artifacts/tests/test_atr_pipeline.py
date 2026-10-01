import json
import math
import shutil
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from artifacts.janus_artifact.atr import (
    ATR_CHECKPOINT_VERSION,
    _checkpoint_contract,
    _contract_sha256,
    _sha256,
    create_synthetic_atr_fixture,
    load_atr_checkpoint,
    predict_atr,
    run_synthetic_atr_smoke,
    train_atr,
)
from artifacts.janus_artifact.atr_data import load_atr_records
from artifacts.janus_artifact.atr_features import ATRConfig
from artifacts.janus_artifact.qai import QAIResNet18
from artifacts.janus_artifact.schema import ValidationError


class ATRPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture.name)
        cls.records = load_atr_records(create_synthetic_atr_fixture(cls.root / "data"))
        cls.config = ATRConfig(base_channels=2, batch_size=4, epochs=1, seed=19)
        cls.training = train_atr(
            cls.records, cls.config, cls.root / "checkpoint", device="cpu",
            run_kind="synthetic_smoke_only",
        )

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.checkpoint = Path(self.work.name) / "checkpoint"
        shutil.copytree(self.root / "checkpoint", self.checkpoint)

    def metadata(self):
        return json.loads((self.checkpoint / "metadata.json").read_text())

    def envelope(self):
        # Only tensors produced by this test's synthetic training are loaded.
        return torch.load(self.checkpoint / "model_state.pt", map_location="cpu", weights_only=True)

    def write_metadata(self, metadata):
        (self.checkpoint / "metadata.json").write_text(json.dumps(metadata) + "\n")

    def rewrite_envelope(self, envelope, metadata):
        torch.save(envelope, self.checkpoint / "model_state.pt")
        metadata["weights_sha256"] = _sha256(self.checkpoint / "model_state.pt")
        self.write_metadata(metadata)

    def rebind(self, metadata):
        envelope = self.envelope()
        envelope["contract"] = _checkpoint_contract(metadata)
        envelope["contract_sha256"] = _contract_sha256(envelope["contract"])
        metadata["contract_sha256"] = envelope["contract_sha256"]
        self.rewrite_envelope(envelope, metadata)

    def renamed_train_as_test(self):
        record = deepcopy(next(row for row in self.records if row["split"] == "train"))
        record["split"] = "test"
        record["response_id"] = "new-test-response"
        record["case_id"] = "new-test-case"
        for step in record["steps"]:
            step["step_id"] = f"new-test-step-{step['step_index']}"
        return record

    def test_training_updates_and_freezes_only_train_candidates(self):
        self.assertTrue(self.training["gradient_observed"])
        self.assertTrue(self.training["parameter_changed"])
        self.assertEqual(self.training["train_steps"], 24)
        self.assertEqual(self.training["candidate_token_ids"], [11, 17])
        self.assertNotIn(23, self.training["candidate_token_ids"])
        self.assertFalse(self.training["scientific_result"])
        self.assertTrue(math.isfinite(self.training["history"][0]["train_loss"]))

    def test_safe_reload_binds_metadata_and_state(self):
        original_load = torch.load
        with patch("artifacts.janus_artifact.atr.torch.load", wraps=original_load) as safe_load:
            model, metadata, device = load_atr_checkpoint(self.checkpoint, device="cpu")
        self.assertTrue(safe_load.call_args.kwargs["weights_only"])
        self.assertEqual(metadata["checkpoint_version"], ATR_CHECKPOINT_VERSION)
        self.assertEqual(str(device), "cpu")
        self.assertFalse(model.training)
        self.assertEqual(metadata["contract_sha256"], _contract_sha256(_checkpoint_contract(metadata)))

    def test_inference_preserves_every_gold_token_and_missing_step(self):
        predictions, metric = predict_atr(self.records, self.checkpoint, device="cpu")
        self.assertEqual(len(predictions), 6)
        self.assertEqual(metric["micro_denominator_all_gold_tokens"], 6)
        self.assertEqual(metric["out_of_vocabulary_gold_tokens_counted_incorrect"], 1)
        self.assertEqual(metric["missing_predictions"], 1)
        self.assertEqual(metric["candidate_coverage"], 5 / 6)
        skipped = [row for row in predictions if row["classification_skipped"] == "missing_profile"]
        self.assertEqual(len(skipped), 1)
        self.assertIsNone(skipped[0]["predicted_token_id"])
        self.assertIsNone(skipped[0]["predicted_probability"])
        self.assertTrue(all(row["predicted_token_id"] in (None, 11, 17) for row in predictions))

    def test_manifest_and_step_order_do_not_change_inference(self):
        baseline, _ = predict_atr(self.records, self.checkpoint, device="cpu")
        records = deepcopy(list(reversed(self.records)))
        for record in records:
            record["steps"].reverse()
        actual, _ = predict_atr(records, self.checkpoint, device="cpu")
        self.assertEqual(actual, baseline)

    def test_unlabeled_prediction_is_allowed_but_evaluation_rejected(self):
        records = deepcopy(self.records)
        for record in records:
            if record["split"] == "test":
                for step in record["steps"]:
                    step["gold_token_id"] = None
        predictions, metric = predict_atr(records, self.checkpoint, device="cpu", evaluate=False)
        self.assertEqual(len(predictions), 6)
        self.assertIsNone(metric)
        with self.assertRaisesRegex(ValidationError, "requires gold token"):
            predict_atr(records, self.checkpoint, device="cpu")

    def test_all_missing_response_profiles_remain_in_denominator(self):
        records = deepcopy([row for row in self.records if row["split"] == "test"])
        for step in records[0]["steps"]:
            step["profile"] = None
        predictions, metric = predict_atr(records, self.checkpoint, device="cpu")
        self.assertEqual(len(predictions), 6)
        self.assertEqual(metric["missing_predictions"], 4)
        self.assertEqual(metric["micro_denominator_all_gold_tokens"], 6)

    def test_train_response_reclassified_test_is_rejected(self):
        record = deepcopy(next(row for row in self.records if row["split"] == "train"))
        record["split"] = "test"
        with self.assertRaisesRegex(ValidationError, "provenance overlap"):
            predict_atr([record], self.checkpoint, device="cpu")

    def test_train_profiles_with_new_response_case_step_ids_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, "provenance overlap"):
            predict_atr([self.renamed_train_as_test()], self.checkpoint, device="cpu")

    def test_float32_rounding_alias_with_all_new_ids_is_rejected(self):
        record = self.renamed_train_as_test()
        for step in record["steps"]:
            step["gold_token_id"] = 23
            for layer in step["profile"]:
                for head in layer:
                    for index, value in enumerate(head):
                        head[index] = value + 1e-11
        with self.assertRaisesRegex(ValidationError, "provenance overlap"):
            predict_atr([record], self.checkpoint, device="cpu")

    def test_integer_float_and_signed_zero_profile_aliases_rejected(self):
        records = deepcopy([row for row in self.records if row["split"] == "train"][:2])
        for record in records:
            for step in record["steps"]:
                for layer in step["profile"]:
                    for head in layer:
                        for index, value in enumerate(head):
                            head[index] = int(round(value))
        directory = Path(self.work.name) / "integer-profiles"
        train_atr(records, self.config, directory, device="cpu")
        for negative_zero in (False, True):
            with self.subTest(negative_zero=negative_zero):
                record = deepcopy(records[0])
                record.update(response_id="alias-test-response", case_id="alias-test-case", split="test")
                for step in record["steps"]:
                    step["step_id"] = f"alias-test-step-{step['step_index']}"
                    step["gold_token_id"] = 23
                    for layer in step["profile"]:
                        for head in layer:
                            for index, value in enumerate(head):
                                head[index] = -0.0 if negative_zero and value == 0 else float(value)
                with self.assertRaisesRegex(ValidationError, "provenance overlap"):
                    predict_atr([record], directory, device="cpu")

    def test_training_case_overlap_is_rejected_across_manifests(self):
        record = deepcopy(next(row for row in self.records if row["split"] == "test"))
        record["case_id"] = next(row for row in self.records if row["split"] == "train")["case_id"]
        with self.assertRaisesRegex(ValidationError, "provenance overlap"):
            predict_atr([record], self.checkpoint, device="cpu")

    def test_training_step_overlap_is_rejected_across_manifests(self):
        record = deepcopy(next(row for row in self.records if row["split"] == "test"))
        record["steps"][0]["step_id"] = next(row for row in self.records if row["split"] == "train")["steps"][0]["step_id"]
        with self.assertRaisesRegex(ValidationError, "provenance overlap"):
            predict_atr([record], self.checkpoint, device="cpu")

    def test_known_train_gold_or_profile_changes_are_rejected(self):
        for field in ("gold", "profile"):
            with self.subTest(field=field):
                records = deepcopy(self.records)
                record = next(row for row in records if row["split"] == "train")
                if field == "gold":
                    step = record["steps"][0]
                    step["gold_token_id"] = 17 if step["gold_token_id"] == 11 else 11
                else:
                    record["steps"][0]["profile"][0][0][0] += 0.2
                with self.assertRaisesRegex(ValidationError, "differs from checkpoint"):
                    predict_atr(records, self.checkpoint, device="cpu")

    def test_overlap_is_checked_in_unselected_split(self):
        records = deepcopy([row for row in self.records if row["split"] == "test"])
        leaked = self.renamed_train_as_test()
        leaked["split"] = "validation"
        records.append(leaked)
        with self.assertRaisesRegex(ValidationError, "provenance overlap"):
            predict_atr(records, self.checkpoint, device="cpu", split="test")

    def test_contract_mismatch_including_unselected_records_is_rejected(self):
        records = deepcopy(self.records)
        record = next(row for row in records if row["split"] == "validation")
        record["tokenizer"]["revision"] = "another-explicit-revision"
        with self.assertRaises(ValidationError):
            predict_atr(records, self.checkpoint, device="cpu")

    def test_single_selected_contract_mismatch_is_rejected(self):
        records = deepcopy([row for row in self.records if row["split"] == "test"])
        for record in records:
            record["tokenizer"]["revision"] = "another-explicit-revision"
        with self.assertRaisesRegex(ValidationError, "does not match checkpoint"):
            predict_atr(records, self.checkpoint, device="cpu")

    def typed_metadata_checkpoint(self):
        records = deepcopy(self.records)
        for record in records:
            record["feature_contract"]["reconstruction"]["parameters"] = {
                "flag": True, "nested": {"weight": 1},
            }
            record["provenance"]["evidence_parameters"] = {
                "known_alignment": True, "score": 1,
            }
        directory = Path(self.work.name) / "typed-metadata"
        train_atr(records, self.config, directory, device="cpu")
        return records, directory

    def test_checkpoint_contract_rejects_boolean_numeric_and_integer_float_aliases(self):
        records, directory = self.typed_metadata_checkpoint()
        for field, value in (("flag", 1), ("flag", 1.0), ("weight", 1.0)):
            with self.subTest(field=field, value=value):
                changed = deepcopy(records)
                for record in changed:
                    parameters = record["feature_contract"]["reconstruction"]["parameters"]
                    if field == "flag":
                        parameters["flag"] = value
                    else:
                        parameters["nested"]["weight"] = value
                with self.assertRaisesRegex(ValidationError, "does not match checkpoint"):
                    predict_atr(changed, directory, device="cpu")

    def test_checkpoint_train_provenance_rejects_primitive_type_aliases(self):
        records, directory = self.typed_metadata_checkpoint()
        for field, value in (("known_alignment", 1), ("known_alignment", 1.0), ("score", 1.0)):
            with self.subTest(field=field, value=value):
                changed = deepcopy(records)
                record = next(row for row in changed if row["split"] == "train")
                record["provenance"]["evidence_parameters"][field] = value
                with self.assertRaisesRegex(ValidationError, "differs from checkpoint"):
                    predict_atr(changed, directory, device="cpu")

    def test_checkpoint_contract_accepts_equivalent_json_object_key_order(self):
        records, directory = self.typed_metadata_checkpoint()
        for record in records:
            parameters = record["feature_contract"]["reconstruction"]["parameters"]
            record["feature_contract"]["reconstruction"]["parameters"] = dict(reversed(list(parameters.items())))
            evidence = record["provenance"]["evidence_parameters"]
            record["provenance"]["evidence_parameters"] = dict(reversed(list(evidence.items())))
        predictions, metric = predict_atr(records, directory, device="cpu")
        self.assertEqual(len(predictions), 6)
        self.assertEqual(metric["micro_denominator_all_gold_tokens"], 6)

    def test_metadata_change_is_not_bound_to_old_weights(self):
        metadata = self.metadata()
        metadata["config"]["augmentation_strength"] = 0.25
        metadata["contract_sha256"] = _contract_sha256(_checkpoint_contract(metadata))
        self.write_metadata(metadata)
        with self.assertRaisesRegex(ValidationError, "metadata/state contract mismatch"):
            load_atr_checkpoint(self.checkpoint, device="cpu")

    def test_rebound_semantic_mismatches_are_rejected(self):
        for field in ("architecture", "token_width", "spatial_shape", "source", "alignment"):
            with self.subTest(field=field):
                metadata = self.metadata()
                if field == "architecture":
                    metadata["architecture"] = "other"
                elif field == "token_width":
                    metadata["config"]["token_width"] = 5
                elif field == "spatial_shape":
                    metadata["input_shape"][1] = 5
                elif field == "source":
                    metadata["training_source_kinds"] = ["reconstructed_recorded"]
                else:
                    metadata["record_contract"]["alignment"] = "invalid"
                self.rebind(metadata)
                with self.assertRaises(ValidationError):
                    load_atr_checkpoint(self.checkpoint, device="cpu")
                shutil.copyfile(self.root / "checkpoint" / "metadata.json", self.checkpoint / "metadata.json")
                shutil.copyfile(self.root / "checkpoint" / "model_state.pt", self.checkpoint / "model_state.pt")

    def test_duplicate_and_reordered_candidates_are_rejected_when_rebound(self):
        for ids in ([11, 11], [17, 11]):
            with self.subTest(ids=ids):
                metadata = self.metadata()
                vocab = metadata["candidate_vocabulary"]
                vocab["token_ids"] = ids
                vocab["sha256"] = _contract_sha256({key: value for key, value in vocab.items() if key != "sha256"})
                self.rebind(metadata)
                with self.assertRaisesRegex(ValidationError, "unique and numerically sorted"):
                    load_atr_checkpoint(self.checkpoint, device="cpu")

    def test_provenance_vocabulary_mismatch_rejected_when_rebound(self):
        metadata = self.metadata()
        for row in metadata["training_provenance"]["records"]:
            row["gold_token_ids"] = [11] * len(row["gold_token_ids"])
        self.rebind(metadata)
        with self.assertRaisesRegex(ValidationError, "vocabulary/provenance mismatch"):
            load_atr_checkpoint(self.checkpoint, device="cpu")

    def test_nonfinite_state_is_rejected_with_matching_file_digest(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                envelope = self.envelope()
                envelope["state_dict"]["classifier.weight"][0, 0] = value
                self.rewrite_envelope(envelope, self.metadata())
                with self.assertRaisesRegex(ValidationError, "non-finite"):
                    load_atr_checkpoint(self.checkpoint, device="cpu")

    def test_state_shape_or_dtype_mismatch_is_rejected(self):
        original = self.envelope()
        for mutation in ("shape", "dtype"):
            with self.subTest(mutation=mutation):
                envelope = deepcopy(original)
                state = envelope["state_dict"]
                state["classifier.weight"] = state["classifier.weight"][:1] if mutation == "shape" else state["classifier.weight"].double()
                self.rewrite_envelope(envelope, self.metadata())
                with self.assertRaisesRegex(ValidationError, "state/config contract mismatch"):
                    load_atr_checkpoint(self.checkpoint, device="cpu")

    def test_nonfinite_inference_logits_and_probabilities_rejected(self):
        with patch("artifacts.janus_artifact.atr.QAIResNet18.forward", lambda model, inputs: torch.full((len(inputs), 2), float("nan"), device=inputs.device)):
            with self.assertRaisesRegex(ValidationError, "inference logits"):
                predict_atr(self.records, self.checkpoint, device="cpu")
        with patch("artifacts.janus_artifact.atr.torch.softmax", lambda logits, dim: torch.full_like(logits, float("inf"))):
            with self.assertRaisesRegex(ValidationError, "inference probabilities"):
                predict_atr(self.records, self.checkpoint, device="cpu")

    def test_nonfinite_training_logits_rejected_before_save(self):
        with patch("artifacts.janus_artifact.atr.QAIResNet18.forward", lambda model, inputs: torch.full((len(inputs), 2), float("nan"), device=inputs.device)):
            with self.assertRaisesRegex(ValidationError, "training logits"):
                train_atr(self.records, self.config, Path(self.work.name) / "bad-logits", device="cpu")
        self.assertFalse((Path(self.work.name) / "bad-logits" / "metadata.json").exists())

    def test_nonfinite_validation_logits_rejected(self):
        original = QAIResNet18.forward
        def forward(model, inputs):
            return original(model, inputs) if model.training else torch.full((len(inputs), 2), float("inf"), device=inputs.device)
        with patch("artifacts.janus_artifact.atr.QAIResNet18.forward", forward):
            with self.assertRaisesRegex(ValidationError, "validation logits"):
                train_atr(self.records, self.config, Path(self.work.name) / "bad-validation", device="cpu")

    def test_nonfinite_gradient_rejected(self):
        original = QAIResNet18
        def model_factory(*args, **kwargs):
            model = original(*args, **kwargs)
            next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
            return model
        with patch("artifacts.janus_artifact.atr.QAIResNet18", model_factory):
            with self.assertRaisesRegex(ValidationError, "training gradient"):
                train_atr(self.records, self.config, Path(self.work.name) / "bad-gradient", device="cpu")

    def test_train_only_and_train_plus_test_without_validation_are_supported(self):
        for include_test in (False, True):
            with self.subTest(include_test=include_test):
                records = [row for row in self.records if row["split"] == "train" or include_test and row["split"] == "test"]
                report = train_atr(records, self.config, Path(self.work.name) / f"no-validation-{include_test}", device="cpu")
                self.assertEqual(report["history"][0]["train_total"], 24)
                self.assertEqual(report["validation_coverage"]["responses"], 0)
                self.assertIsNone(report["validation_coverage"]["candidate_coverage"])
                self.assertNotIn("validation_loss", report["history"][0])

    def test_validation_oov_is_reported_and_only_known_steps_enter_ce(self):
        records = deepcopy(self.records)
        val = next(row for row in records if row["split"] == "validation")
        val["steps"][0]["gold_token_id"] = 23
        report = train_atr(records, self.config, Path(self.work.name) / "val-oov", device="cpu")
        self.assertEqual(report["validation_coverage"]["gold_tokens"], 6)
        self.assertEqual(report["validation_coverage"]["oov_gold_tokens"], 1)
        self.assertEqual(report["validation_coverage"]["cross_entropy_steps"], 5)
        self.assertEqual(report["history"][0]["validation_total"], 5)

    def _train33_records(self, grid_size=4):
        records = deepcopy([row for row in self.records if row["split"] == "train"])
        for index in range(3):
            record = deepcopy(records[index])
            record["response_id"] = f"added-train-response-{index}"
            record["case_id"] = f"added-train-case-{index}"
            for step in record["steps"]:
                step["step_id"] = f"added-train-{index}-step-{step['step_index']}"
                step["profile"][0][0][0] += 0.001 * (index + 1)
            records.append(record)
        if grid_size != 4:
            for record in records:
                contract = record["feature_contract"]
                contract["layer_ids"] = [f"layer-{index}" for index in range(grid_size)]
                contract["kv_head_ids"] = [f"head-{index}" for index in range(grid_size)]
                for step in record["steps"]:
                    original = step["profile"]
                    step["profile"] = [
                        [deepcopy(original[layer % 4][head % 4]) for head in range(grid_size)]
                        for layer in range(grid_size)
                    ]
        return records

    def test_train33_batch32_retains_every_step(self):
        report = train_atr(
            self._train33_records(), replace(self.config, batch_size=32),
            Path(self.work.name) / "train33", device="cpu",
        )
        self.assertEqual(report["model_input_shape"], [4, 4, 4])
        self.assertEqual(report["train_steps"], 33)
        self.assertTrue(all(epoch["train_total"] == 33 for epoch in report["history"]))
        self.assertTrue(report["gradient_observed"])
        self.assertTrue(report["parameter_changed"])

    def test_train33_batch32_on_32_by_32_grid_retains_every_step(self):
        report = train_atr(
            self._train33_records(grid_size=32),
            replace(self.config, batch_size=32, epochs=2),
            Path(self.work.name) / "train33-grid32", device="cpu",
        )
        self.assertEqual(report["model_input_shape"], [4, 32, 32])
        self.assertEqual(report["train_steps"], 33)
        self.assertEqual(len(report["history"]), 2)
        self.assertTrue(all(epoch["train_total"] == 33 for epoch in report["history"]))
        self.assertTrue(report["gradient_observed"])
        self.assertTrue(report["parameter_changed"])

    def test_batch_one_rejected_before_reading_records(self):
        with self.assertRaisesRegex(ValidationError, "batch_size must be at least two"):
            train_atr([], replace(self.config, batch_size=1), Path(self.work.name) / "batch1", device="cpu")

    def test_test_feature_width_does_not_influence_training(self):
        records = deepcopy(self.records)
        for record in records:
            if record["split"] == "test":
                for step in record["steps"]:
                    if step["profile"] is not None:
                        for layer in step["profile"]:
                            for head in layer:
                                head.append(0.1)
        report = train_atr(records, self.config, Path(self.work.name) / "test-width", device="cpu")
        self.assertEqual(report["model_input_shape"], [4, 4, 4])
        with self.assertRaisesRegex(ValidationError, "exceeds train-derived width"):
            predict_atr(records, Path(self.work.name) / "test-width", device="cpu")

    def test_unselected_feature_width_does_not_influence_inference(self):
        records = deepcopy(self.records)
        for record in records:
            if record["split"] == "validation":
                for step in record["steps"]:
                    for layer in step["profile"]:
                        for head in layer:
                            head.append(0.1)
        predictions, metric = predict_atr(records, self.checkpoint, device="cpu", split="test")
        self.assertEqual(len(predictions), 6)
        self.assertEqual(metric["micro_denominator_all_gold_tokens"], 6)

    def test_synthetic_smoke_report_asserts_gradient_reload_and_denominator(self):
        report = run_synthetic_atr_smoke(Path(self.work.name) / "smoke", device="cpu")
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["checkpoint_reload_verified"])
        self.assertFalse(report["scientific_result"])
        self.assertEqual(report["dasr_smoke_only"]["micro_denominator_all_gold_tokens"], 6)


if __name__ == "__main__":
    unittest.main()
