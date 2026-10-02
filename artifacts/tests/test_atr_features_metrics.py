"""Synthetic causal-feature and complete-denominator ATR regressions."""

import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch

from artifacts.janus_artifact.atr_data import freeze_atr_vocabulary
from artifacts.janus_artifact.atr_features import (
    ATRConfig, ATRTokenDataset, derive_atr_input_shape, prepare_atr_features,
)
from artifacts.janus_artifact.atr_metrics import compute_atr_dasr
from artifacts.janus_artifact.metrics import compute_dasr
from artifacts.janus_artifact.schema import ValidationError
from artifacts.tests.atr_fixture_helpers import synthetic_response


class ATRFeatureTests(unittest.TestCase):
    def test_causal_running_mean_preserves_dimensions_and_exact_history(self):
        record = synthetic_response()
        shape = derive_atr_input_shape([record], ATRConfig())
        raw = prepare_atr_features([record], ATRConfig(sequential_augmentation="none"), shape)
        augmented = prepare_atr_features([record], ATRConfig(augmentation_strength=0.5), shape)
        keys = [(record["response_id"], step["step_id"]) for step in record["steps"]]
        self.assertEqual(tuple(augmented[keys[2]].shape), shape)
        self.assertTrue(torch.equal(augmented[keys[0]], raw[keys[0]]))
        self.assertTrue(torch.allclose(augmented[keys[1]], raw[keys[1]] + 0.5 * raw[keys[0]]))
        self.assertTrue(torch.allclose(
            augmented[keys[2]], raw[keys[2]] + 0.25 * (raw[keys[0]] + raw[keys[1]])
        ))

    def test_future_profiles_and_gold_tokens_do_not_change_prior_features(self):
        record = synthetic_response()
        changed = copy.deepcopy(record)
        changed["steps"][2]["profile"][0][0][0] += 1000
        for step in changed["steps"]:
            step["gold_token_id"] = 17 if step["gold_token_id"] == 11 else 11
        shape = derive_atr_input_shape([record], ATRConfig())
        first = prepare_atr_features([record], ATRConfig(), shape)
        second = prepare_atr_features([changed], ATRConfig(), shape)
        for step in record["steps"][:2]:
            key = (record["response_id"], step["step_id"])
            self.assertTrue(torch.equal(first[key], second[key]))

    def test_step_list_order_does_not_change_causal_features(self):
        record = synthetic_response()
        shuffled = copy.deepcopy(record)
        shuffled["steps"].reverse()
        shape = derive_atr_input_shape([record], ATRConfig())
        ordered = prepare_atr_features([record], ATRConfig(), shape)
        reversed_rows = prepare_atr_features([shuffled], ATRConfig(), shape)
        self.assertTrue(all(torch.equal(ordered[key], reversed_rows[key]) for key in ordered))

    def test_history_resets_between_responses(self):
        first = synthetic_response("a")
        second = synthetic_response("b", offset=100)
        features = prepare_atr_features([first, second], ATRConfig(), (3, 2, 2))
        alone = prepare_atr_features([second], ATRConfig(), (3, 2, 2))
        self.assertTrue(all(
            torch.equal(value, alone[key]) for key, value in features.items() if key[0] == "b"
        ))

    def test_missing_step_is_skipped_without_changing_later_step_alignment(self):
        record = synthetic_response(split="test")
        record["steps"][1]["profile"] = None
        features = prepare_atr_features([record], ATRConfig(), (3, 2, 2))
        raw = prepare_atr_features([record], ATRConfig(sequential_augmentation="none"), (3, 2, 2))
        keys = [(record["response_id"], step["step_id"]) for step in record["steps"]]
        self.assertNotIn(keys[1], features)
        self.assertTrue(torch.allclose(features[keys[2]], raw[keys[2]] + 0.5 * raw[keys[0]]))

    def test_token_width_train_only_and_longer_inference_rejected(self):
        train = synthetic_response()
        test = synthetic_response("heldout", split="test")
        for step in test["steps"]:
            for layer in step["profile"]:
                for head in layer:
                    head.extend([4.0, 5.0])
        shape = derive_atr_input_shape([train], ATRConfig())
        self.assertEqual(shape, (3, 2, 2))
        with self.assertRaisesRegex(ValidationError, "train records only"):
            derive_atr_input_shape([train, test], ATRConfig())
        with self.assertRaisesRegex(ValidationError, "train-derived width"):
            prepare_atr_features([test], ATRConfig(), shape)
        with self.assertRaisesRegex(ValidationError, "truncate"):
            derive_atr_input_shape([train], ATRConfig(token_width=2))

    def test_conflicting_special_token_alignment_is_rejected(self):
        record = synthetic_response()
        record["alignment"]["special_tokens"] = "excluded"
        with self.assertRaisesRegex(ValidationError, "BOS/EOS"):
            prepare_atr_features([record], ATRConfig(), (3, 2, 2))

    def test_explicit_normalization_and_padding(self):
        record = synthetic_response()
        features = prepare_atr_features(
            [record], ATRConfig(normalization="per_layer_head_minmax", sequential_augmentation="none"),
            (5, 2, 2),
        )
        feature = features[(record["response_id"], record["steps"][0]["step_id"])]
        self.assertEqual(tuple(feature.shape), (5, 2, 2))
        self.assertEqual(float(feature.min()), 0)
        self.assertEqual(float(feature.max()), 1)
        self.assertEqual(int(torch.count_nonzero(feature[3:])), 0)

    def test_float32_overflow_is_rejected_before_model_use(self):
        record = synthetic_response()
        record["steps"][0]["profile"][0][0][0] = 1e300
        with self.assertRaisesRegex(ValidationError, "float32"):
            prepare_atr_features([record], ATRConfig(), (3, 2, 2))

    def test_token_dataset_excludes_oov_and_missing_but_retains_ids(self):
        train = synthetic_response()
        vocabulary = freeze_atr_vocabulary([train])
        test = synthetic_response("heldout", split="test")
        test["steps"][0]["gold_token_id"] = 23
        test["steps"][1]["profile"] = None
        features = prepare_atr_features([test], ATRConfig(), (3, 2, 2))
        dataset = ATRTokenDataset(
            [test], features, {token: index for index, token in enumerate(vocabulary["token_ids"])}
        )
        self.assertEqual(len(dataset), 1)
        _, target, response_id, step_id, step_index, token = dataset[0]
        self.assertEqual((response_id, step_id, step_index, token),
                         ("heldout", "heldout-step-2", 2, 11))
        self.assertEqual(int(target), 0)

    def test_empty_token_dataset_is_valid_without_relaxing_manifest_validation(self):
        self.assertEqual(len(ATRTokenDataset([], {}, {11: 0, 17: 1})), 0)

    def test_config_rejects_ambiguous_and_unknown_choices(self):
        for config in (
            ATRConfig(batch_size=1), ATRConfig(sequential_augmentation="future_mean"),
            ATRConfig(augmentation_strength=math.nan), ATRConfig(num_workers=True),
            ATRConfig(seed=-1), ATRConfig(normalization="dataset_test_fit"),
            ATRConfig(learning_rate=10**1000),
        ):
            with self.subTest(config=config), self.assertRaises(ValidationError):
                config.validate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"unknown":1}')
            with self.assertRaisesRegex(ValidationError, "unknown"):
                ATRConfig.from_json(path)


class ATRMetricTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = freeze_atr_vocabulary([synthetic_response()])

    def prediction(self, record, index, token):
        step = record["steps"][index]
        return {"response_id": record["response_id"], "step_id": step["step_id"],
                "step_index": index, "predicted_token_id": token}

    def test_step_id_alignment_preserves_middle_missing_and_later_correct(self):
        record = synthetic_response("test", split="test")
        predictions = [self.prediction(record, 2, 11), self.prediction(record, 0, 11)]
        result = compute_atr_dasr([record], predictions, self.vocabulary)
        self.assertEqual(result["micro_correct"], 2)
        self.assertEqual(result["micro_denominator_all_gold_tokens"], 3)
        self.assertEqual(result["missing_predictions"], 1)
        self.assertAlmostEqual(result["dasr_macro"], 2 / 3)

    def test_oov_and_missing_remain_in_complete_denominator(self):
        record = synthetic_response("test", split="test")
        record["steps"][2]["gold_token_id"] = 23
        predictions = [self.prediction(record, 0, 11), self.prediction(record, 2, 17)]
        result = compute_atr_dasr([record], predictions, self.vocabulary)
        self.assertEqual(result["micro_correct"], 1)
        self.assertEqual(result["micro_denominator_all_gold_tokens"], 3)
        self.assertEqual(result["missing_predictions"], 1)
        self.assertEqual(result["out_of_vocabulary_gold_tokens_counted_incorrect"], 1)
        self.assertAlmostEqual(result["candidate_coverage"], 2 / 3)

    def test_response_macro_average_is_distinct_from_token_micro_average(self):
        short = synthetic_response("short", split="test", count=1)
        long = synthetic_response("long", split="test", count=3, offset=3)
        predictions = [self.prediction(short, 0, 11), self.prediction(long, 1, 17)]
        result = compute_atr_dasr([long, short], predictions, self.vocabulary)
        self.assertAlmostEqual(result["dasr_macro"], 2 / 3)
        self.assertEqual(result["micro_rate"], 0.5)
        self.assertEqual(result["missing_predictions"], 2)

    def test_unknown_duplicate_mismatched_and_nonfinite_predictions_rejected(self):
        record = synthetic_response("test", split="test")
        valid = self.prediction(record, 0, 11)
        cases = [
            [valid, valid],
            [{**valid, "step_id": "unknown"}],
            [{**valid, "step_index": 1}],
            [{**valid, "predicted_token_id": 23}],
            [{**valid, "predicted_token_id": True}],
            [{**valid, "predicted_probability": math.inf}],
            [{**valid, "predicted_probability": 10**1000}],
        ]
        for predictions in cases:
            with self.subTest(predictions=predictions), self.assertRaises(ValidationError):
                compute_atr_dasr([record], predictions, self.vocabulary)

    def test_evaluation_rejects_missing_gold_alignment(self):
        record = synthetic_response("test", split="test")
        record["steps"][1]["gold_token_id"] = None
        with self.assertRaises(ValidationError):
            compute_atr_dasr([record], [], self.vocabulary)

    def test_existing_dasr_counts_internal_null_without_shifting(self):
        result = compute_dasr([{"response_id": "r", "gold_tokens": ["A", "B", "C"],
                               "predicted_tokens": ["A", None, "C"]}])
        self.assertEqual(result["micro_correct"], 2)
        self.assertEqual(result["missing_predictions"], 1)
        self.assertEqual(result["micro_denominator_all_gold_tokens"], 3)
