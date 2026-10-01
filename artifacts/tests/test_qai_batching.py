"""Regression coverage for training batches that retain every synthetic sample."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from artifacts.janus_artifact import qai
from artifacts.janus_artifact.schema import ValidationError


ARTIFACTS_ROOT = Path(__file__).resolve().parents[1]


class QAIBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_training_batches_retain_every_index_without_singletons(self):
        for sample_count, batch_size, expected_sizes in (
            (2, 32, [2]),
            (32, 32, [32]),
            (33, 32, [33]),
            (65, 32, [32, 33]),
            (3, 2, [3]),
        ):
            with self.subTest(sample_count=sample_count, batch_size=batch_size):
                batches = qai._training_batch_indices(
                    sample_count, batch_size, torch.Generator().manual_seed(19)
                )
                self.assertEqual([len(batch) for batch in batches], expected_sizes)
                flattened = [index for batch in batches for index in batch]
                self.assertEqual(sorted(flattened), list(range(sample_count)))
                self.assertEqual(len(flattened), len(set(flattened)))

    def test_training_batch_order_is_seeded_and_advances_each_epoch(self):
        first_generator = torch.Generator().manual_seed(19)
        first_epoch = qai._training_batch_indices(65, 32, first_generator)
        second_epoch = qai._training_batch_indices(65, 32, first_generator)
        repeat_generator = torch.Generator().manual_seed(19)
        self.assertEqual(
            first_epoch, qai._training_batch_indices(65, 32, repeat_generator)
        )
        self.assertEqual(
            second_epoch, qai._training_batch_indices(65, 32, repeat_generator)
        )
        self.assertNotEqual(first_epoch, second_epoch)

    def test_batch_size_one_is_rejected_before_reading_payloads(self):
        records = [
            qai.QAIRecord(
                sample_id=f"unread-sample-{index}",
                case_id=f"unread-case-{index}",
                split="train",
                attribute="synthetic-topic",
                attribute_present=True,
                label=f"label-{index}",
                npz_path="must-not-be-opened.npz",
            )
            for index in range(2)
        ]
        with patch.object(
            qai, "prefill_rank_feature", side_effect=AssertionError("payload was read")
        ):
            with self.assertRaisesRegex(ValidationError, "batch_size"):
                qai.train_qai(
                    records,
                    qai.QAIConfig(batch_size=1),
                    ARTIFACTS_ROOT / "must-not-be-created-checkpoint",
                    device="cpu",
                )

    def test_training_33_samples_batch32_on_32_by_32_grid_retains_all_samples(self):
        with tempfile.TemporaryDirectory(
            prefix=".qai-batching-test-", dir=ARTIFACTS_ROOT
        ) as directory:
            root = Path(directory)
            rng = np.random.default_rng(23)
            records = []
            for index in range(33):
                payload = root / f"synthetic-{index}.npz"
                np.savez_compressed(
                    payload,
                    attn_rank=rng.integers(
                        0, 4096, size=(32, 32, 2, 4), dtype=np.uint16
                    ),
                    top_k=np.asarray(4, dtype=np.uint16),
                )
                records.append(
                    qai.QAIRecord(
                        sample_id=f"synthetic-sample-{index}",
                        case_id=f"synthetic-case-{index}",
                        split="train",
                        attribute="synthetic-topic",
                        attribute_present=True,
                        label=f"label-{index % 2}",
                        npz_path=str(payload),
                    )
                )
            batch_sizes = []
            layer4_shapes = []
            original_forward = qai.QAIResNet18.forward

            def observed_forward(model, features):
                if not model.training:
                    return original_forward(model, features)
                batch_sizes.append(int(features.shape[0]))
                handle = model.layer4.register_forward_hook(
                    lambda _module, _inputs, output: layer4_shapes.append(
                        tuple(output.shape)
                    )
                )
                try:
                    return original_forward(model, features)
                finally:
                    handle.remove()

            with patch.object(qai.QAIResNet18, "forward", observed_forward):
                report = qai.train_qai(
                    records,
                    qai.QAIConfig(
                        selected_ranks=1,
                        base_channels=2,
                        batch_size=32,
                        epochs=1,
                        learning_rate=1e-3,
                        seed=19,
                    ),
                    root / "checkpoint",
                    device="cpu",
                    run_kind="synthetic_smoke_only",
                )
            self.assertEqual(batch_sizes, [33])
            self.assertEqual(sum(batch_sizes), 33)
            self.assertEqual(layer4_shapes, [(33, 16, 1, 1)])
            self.assertEqual(report["train_samples"], 33)
            self.assertEqual(report["history"][0]["train_total"], 33)
            self.assertTrue(report["gradient_observed"])
            self.assertTrue(report["parameter_changed"])
            self.assertFalse(report["scientific_result"])


if __name__ == "__main__":
    unittest.main()
