"""Synthetic regressions for training provenance and attribute isolation."""
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from artifacts.janus_artifact.qai import (
    QAIConfig, create_synthetic_qai_fixture, load_qai_records,
    load_qai_checkpoint, predict_qai, train_qai,
)
from artifacts.janus_artifact.schema import ValidationError, write_jsonl


class QAIProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.records = load_qai_records(create_synthetic_qai_fixture(cls.root / "data"))
        cls.checkpoint = cls.root / "checkpoint"
        torch.set_num_threads(1)
        train_qai(
            cls.records,
            QAIConfig(selected_ranks=2, base_channels=2, batch_size=4, epochs=1),
            cls.checkpoint, device="cpu", run_kind="synthetic_regression_only",
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_checkpoint_retains_train_ids_case_ids_and_payload_digests(self):
        _, metadata, _ = load_qai_checkpoint(self.checkpoint, device="cpu")
        rows = metadata["training_provenance"]["records"]
        expected = [r for r in self.records if r.split == "train"]
        self.assertEqual({r["sample_id"] for r in rows}, {r.sample_id for r in expected})
        self.assertEqual({r["case_id"] for r in rows}, {r.case_id for r in expected})
        self.assertTrue(all(
            len(r["payload_sha256"]) == 64 if r["attribute_present"]
            else r["payload_sha256"] is None for r in rows
        ))
        self.assertNotIn(str(self.root), json.dumps(metadata["training_provenance"]))

    def test_independent_manifest_relabels_training_case_as_test(self):
        original = next(r for r in self.records if r.split == "train" and r.attribute_present)
        # Load a separately written manifest so this exercises the public boundary.
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "independent.jsonl"
            write_jsonl(manifest, [{
                "sample_id": "new-id", "case_id": original.case_id, "split": "test",
                "attribute": original.attribute, "attribute_present": True,
                "label": original.label, "npz_path": original.npz_path,
            }])
            with self.assertRaisesRegex(ValidationError, "training provenance overlap"):
                predict_qai(load_qai_records(manifest), self.checkpoint, split="test", device="cpu")

    def test_training_sample_reassigned_with_new_case_is_rejected(self):
        train = next(r for r in self.records if r.split == "train" and r.attribute_present)
        test = next(r for r in self.records if r.split == "test" and r.attribute_present)
        moved = replace(test, sample_id=train.sample_id)
        with self.assertRaisesRegex(ValidationError, "training provenance overlap"):
            predict_qai([moved], self.checkpoint, split="test", device="cpu")

    def test_same_bytes_with_new_sample_and_case_ids_are_rejected(self):
        train = next(r for r in self.records if r.split == "train" and r.attribute_present)
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "copied.npz"
            shutil.copyfile(train.npz_path, copied)
            moved = replace(train, sample_id="copied-id", case_id="copied-case",
                            split="test", npz_path=str(copied))
            with self.assertRaisesRegex(ValidationError, "training provenance overlap"):
                predict_qai([moved], self.checkpoint, split="test", device="cpu")

    def test_duplicate_payload_across_splits_rejected_before_training(self):
        train = next(r for r in self.records if r.split == "train" and r.attribute_present)
        test = next(r for r in self.records if r.split == "test" and r.attribute_present)
        leaked = [replace(r, npz_path=train.npz_path) if r == test else r for r in self.records]
        with patch("artifacts.janus_artifact.qai.QAIResNet18") as model:
            with self.assertRaisesRegex(ValidationError, "payload split leakage"):
                train_qai(leaked, QAIConfig(base_channels=2, epochs=1),
                          self.root / "never-written", device="cpu")
            model.assert_not_called()

    def test_manifest_duplicate_payload_new_ids_across_splits_rejected(self):
        train = next(r for r in self.records if r.split == "train" and r.attribute_present)
        rows = []
        for index, split in enumerate(("train", "test")):
            rows.append({
                "sample_id": f"new-{index}", "case_id": f"case-{index}",
                "split": split, "attribute": train.attribute, "attribute_present": True,
                "label": train.label, "npz_path": train.npz_path,
            })
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "leaked.jsonl"
            write_jsonl(manifest, rows)
            with self.assertRaisesRegex(ValidationError, "payload split leakage"):
                load_qai_records(manifest)

    def test_same_label_for_other_attribute_is_rejected(self):
        test = next(r for r in self.records if r.split == "test" and r.attribute_present)
        with self.assertRaisesRegex(ValidationError, "attribute.*checkpoint"):
            predict_qai([replace(test, attribute="other")], self.checkpoint,
                        split="test", device="cpu")

    def test_mixed_attributes_include_absent_records(self):
        test = next(r for r in self.records if r.split == "test" and r.attribute_present)
        absent = next(r for r in self.records if r.split == "test" and not r.attribute_present)
        with self.assertRaisesRegex(ValidationError, "attribute.*checkpoint"):
            predict_qai([test, replace(absent, attribute="other")], self.checkpoint,
                        split="test", device="cpu")

    def test_absent_only_other_attribute_is_rejected(self):
        absent = next(r for r in self.records if r.split == "test" and not r.attribute_present)
        with self.assertRaisesRegex(ValidationError, "attribute.*checkpoint"):
            predict_qai([replace(absent, attribute="other")], self.checkpoint,
                        split="test", device="cpu")

    def test_absent_only_matching_attribute_has_empty_pasr_denominator(self):
        absent = next(r for r in self.records if r.split == "test" and not r.attribute_present)
        predictions, metrics = predict_qai([absent], self.checkpoint, split="test", device="cpu")
        self.assertEqual(len(predictions), 1)
        self.assertEqual(metrics["micro_denominator_present_queries"], 0)
        self.assertIsNone(predictions[0]["predicted_value"])

    def test_disjoint_independent_test_manifest_is_accepted(self):
        test = [r for r in self.records if r.split == "test"]
        predictions, metrics = predict_qai(test, self.checkpoint, split="test", device="cpu")
        self.assertEqual(len(predictions), 5)
        self.assertEqual(metrics["micro_denominator_present_queries"], 4)

    def test_absent_training_case_reassigned_to_absent_test_is_rejected(self):
        absent = next(r for r in self.records if r.split == "train" and not r.attribute_present)
        moved = replace(absent, sample_id="new-absent-id", split="test")
        with self.assertRaisesRegex(ValidationError, "training provenance overlap"):
            predict_qai([moved], self.checkpoint, split="test", device="cpu")

    def test_nonselected_split_still_checked_against_training_provenance(self):
        train_absent = next(r for r in self.records if r.split == "train" and not r.attribute_present)
        test = next(r for r in self.records if r.split == "test" and r.attribute_present)
        moved = replace(train_absent, sample_id="new-validation-id", split="validation")
        with self.assertRaisesRegex(ValidationError, "training provenance overlap"):
            predict_qai([test, moved], self.checkpoint, split="test", device="cpu")

    def test_duplicate_sample_ids_cannot_overwrite_predictions(self):
        test = next(r for r in self.records if r.split == "test" and r.attribute_present)
        with self.assertRaisesRegex(ValidationError, "duplicate QAI sample_id"):
            predict_qai([test, test], self.checkpoint, split="test", device="cpu")

    def test_checkpoint_without_training_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            metadata = json.loads((self.checkpoint / "metadata.json").read_text())
            metadata.pop("training_provenance")
            (destination / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValidationError, "training provenance"):
                load_qai_checkpoint(destination, device="cpu")
