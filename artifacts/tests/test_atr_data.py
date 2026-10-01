"""Small synthetic regressions for ATR response/step provenance contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from artifacts.janus_artifact.atr_data import (
    assign_atr_grouped_splits, atr_record_contract, atr_record_snapshot,
    freeze_atr_vocabulary, load_atr_records, validate_atr_records,
    validate_atr_vocabulary,
)
from artifacts.janus_artifact.schema import ValidationError


def response(response_id="r0", split="train", *, case_id=None, offset=0.0):
    return {
        "schema_version": "janus.atr.response.v1", "response_id": response_id,
        "case_id": case_id or f"case-{response_id}", "task_id": "synthetic-token-task",
        "split": split, "source_kind": "synthetic_fixture", "provenance": {"synthetic": True},
        "tokenizer": {"name": "synthetic-tokenizer", "revision": "fixture-v1", "vocab_size": 20},
        "alignment": {"step_index_base": 0, "profile_predicts": "same_index_output_token",
                      "bos_included": False, "eos_included": False, "special_tokens": "excluded",
                      "response_scope": "complete_response"},
        "feature_contract": {
            "data_stage": "reconstructed_token_sparsity", "axis_order": ["layer", "kv_head", "key_token"],
            "layer_ids": ["layer-0", "layer-1"], "kv_head_ids": ["head-0", "head-1"],
            "key_position_policy": "absolute_zero_based_prefix_positions",
            "reconstruction": {"method": "synthetic-explicit", "revision": "fixture-v1", "parameters": {}},
        },
        "steps": [
            {"step_id": f"{response_id}-s0", "step_index": 0, "gold_token_id": 2,
             "profile": [[[offset, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]},
            {"step_id": f"{response_id}-s1", "step_index": 1, "gold_token_id": 10,
             "profile": [[[offset, 1.0, 2.0], [2.0, 3.0, 4.0]], [[4.0, 5.0, 6.0], [6.0, 7.0, 8.0]]]},
        ],
    }


class ATRDataTests(unittest.TestCase):
    def test_returns_independent_canonical_records_and_steps(self):
        first, second = response("a"), response("b", "test", offset=3)
        first["steps"].reverse()
        canonical = validate_atr_records([second, first])
        self.assertEqual([r["response_id"] for r in canonical], ["a", "b"])
        self.assertEqual([s["step_index"] for s in canonical[0]["steps"]], [0, 1])
        canonical[0]["steps"][0]["profile"][0][0][0] = 99
        self.assertEqual(first["steps"][1]["profile"][0][0][0], 0)
        self.assertEqual(first["steps"][0]["step_index"], 1)

    def test_train_only_candidate_ids_do_not_include_test_oov(self):
        train, test = response(), response("test", "test", offset=1)
        test["steps"][0]["gold_token_id"] = 19
        vocabulary = freeze_atr_vocabulary([test, train])
        self.assertEqual(vocabulary["token_ids"], [2, 10])
        self.assertEqual(validate_atr_vocabulary(vocabulary, atr_record_contract(train)), {2, 10})
        self.assertNotIn(19, vocabulary["token_ids"])

    def test_vocabulary_binds_task_tokenizer_and_frozen_train_metadata(self):
        vocabulary = freeze_atr_vocabulary([response()])
        for path, value in (("task_id", "other"), ("source_split", "test"), ("frozen", False)):
            modified = deepcopy(vocabulary)
            modified[path] = value
            with self.subTest(path=path), self.assertRaises(ValidationError):
                validate_atr_vocabulary(modified)
        changed = atr_record_contract(response())
        changed["tokenizer"]["revision"] = "fixture-v2"
        with self.assertRaisesRegex(ValidationError, "contract mismatch"):
            validate_atr_vocabulary(vocabulary, changed)

    def test_vocabulary_rejects_duplicate_reordered_bool_and_out_of_range_ids(self):
        vocabulary = freeze_atr_vocabulary([response()])
        for values in ([2, 2], [10, 2], [False, 2], [2, 20], [2]):
            changed = deepcopy(vocabulary)
            changed["token_ids"] = values
            with self.subTest(values=values), self.assertRaises(ValidationError):
                validate_atr_vocabulary(changed)

    def test_freeze_requires_two_training_classes_and_train_response(self):
        one = response()
        one["steps"][1]["gold_token_id"] = 2
        with self.assertRaisesRegex(ValidationError, "at least two"):
            freeze_atr_vocabulary([one])
        with self.assertRaisesRegex(ValidationError, "without train"):
            freeze_atr_vocabulary([response("t", "test")])

    def test_missing_test_observation_retains_steps_and_gold(self):
        record = response("test", "test")
        record["steps"][1]["profile"] = None
        canonical = validate_atr_records([record], require_gold=True)[0]
        self.assertEqual(len(canonical["steps"]), 2)
        self.assertEqual(canonical["steps"][1]["gold_token_id"], 10)
        self.assertIsNone(canonical["steps"][1]["profile"])
        self.assertEqual(len(atr_record_snapshot([record])[0]["gold_token_ids"]), 2)

    def test_pure_prediction_allows_null_gold_but_evaluation_refuses(self):
        record = response("test", "test")
        record["steps"][0]["gold_token_id"] = None
        self.assertIsNone(validate_atr_records([record])[0]["steps"][0]["gold_token_id"])
        with self.assertRaisesRegex(ValidationError, "gold_token_id is required"):
            validate_atr_records([record], require_gold=True)

    def test_training_always_requires_observations_and_gold(self):
        for field in ("profile", "gold_token_id"):
            record = response()
            record["steps"][0][field] = None
            with self.subTest(field=field), self.assertRaises(ValidationError):
                validate_atr_records([record])

    def test_complete_response_indices_and_ids_are_strict(self):
        for index in (-1, True, 0, 3):
            record = response()
            record["steps"][1]["step_index"] = index
            with self.subTest(index=index), self.assertRaises(ValidationError):
                validate_atr_records([record])
        for field in ("response_id", "case_id", "task_id"):
            record = response()
            record[field] = " "
            with self.subTest(field=field), self.assertRaises(ValidationError):
                validate_atr_records([record])

    def test_duplicate_response_step_and_cross_split_case_ids_rejected(self):
        with self.assertRaisesRegex(ValidationError, "duplicate ATR response_id"):
            validate_atr_records([response(), response()])
        a, b = response(), response("b", offset=4)
        b["steps"][0]["step_id"] = a["steps"][0]["step_id"]
        with self.assertRaisesRegex(ValidationError, "duplicate ATR step_id"):
            validate_atr_records([a, b])
        b = response("b", "test", case_id=a["case_id"], offset=4)
        with self.assertRaisesRegex(ValidationError, "case split leakage"):
            validate_atr_records([a, b])

    def test_identical_full_response_profiles_rejected_across_split_with_new_ids_and_gold(self):
        train, test = response(), response("copy", "test")
        test["steps"][0]["gold_token_id"] = 19
        test["steps"].reverse()
        with self.assertRaisesRegex(ValidationError, "complete response profile split leakage"):
            validate_atr_records([test, train])
        self.assertEqual(atr_record_snapshot([train])[0]["profile_sha256"],
                         atr_record_snapshot([test])[0]["profile_sha256"])

    def test_model_dtype_profile_aliases_cannot_bypass_cross_split_payload_check(self):
        train = response("train")
        for step in train["steps"]:
            step["profile"] = [[[int(value) for value in head] for head in layer] for layer in step["profile"]]
        for alias in ("integer_float_spelling", "float32_rounding", "signed_zero"):
            test = deepcopy(train)
            test.update(response_id=f"copy-{alias}", case_id=f"new-case-{alias}", split="test")
            for step in test["steps"]:
                step["step_id"] = f"copy-{alias}-s{step['step_index']}"
                step["gold_token_id"] = 19
                for layer in step["profile"]:
                    for head in layer:
                        for index, value in enumerate(head):
                            head[index] = float(value)
                            if alias == "float32_rounding" and value:
                                head[index] += 1e-8
                            if alias == "signed_zero" and not value:
                                head[index] = -0.0
            with self.subTest(alias=alias):
                self.assertEqual(atr_record_snapshot([train])[0]["profile_sha256"],
                                 atr_record_snapshot([test])[0]["profile_sha256"])
                with self.assertRaisesRegex(ValidationError, "complete response profile split leakage"):
                    validate_atr_records([train, test])

    def test_finite_json_value_that_overflows_model_dtype_is_rejected(self):
        record = response()
        record["steps"][0]["profile"][0][0][0] = 1e308
        with self.assertRaisesRegex(ValidationError, "finite after float32 conversion"):
            validate_atr_records([record])

    def test_model_dtype_profile_digest_binds_missing_positions_and_shapes(self):
        observed = response("observed", "test")
        first_missing, last_missing = deepcopy(observed), deepcopy(observed)
        first_missing["steps"][0]["profile"] = None
        last_missing["steps"][1]["profile"] = None
        self.assertNotEqual(atr_record_snapshot([first_missing])[0]["profile_sha256"],
                            atr_record_snapshot([last_missing])[0]["profile_sha256"])
        reshaped = deepcopy(observed)
        reshaped["feature_contract"]["layer_ids"] = ["layer-0"]
        reshaped["feature_contract"]["kv_head_ids"] = ["head-0", "head-1", "head-2", "head-3"]
        for step in reshaped["steps"]:
            step["profile"] = [[head for layer in step["profile"] for head in layer]]
        self.assertNotEqual(atr_record_snapshot([observed])[0]["profile_sha256"],
                            atr_record_snapshot([reshaped])[0]["profile_sha256"])

    def test_same_single_step_is_allowed_and_missing_profiles_are_not_duplicate_payload(self):
        train, test = response(), response("test", "test", offset=4)
        test["steps"][0]["profile"] = deepcopy(train["steps"][0]["profile"])
        self.assertEqual(len(validate_atr_records([train, test])), 2)
        a, b = response("a", "test"), response("b", "validation")
        for record in (a, b):
            for step in record["steps"]:
                step["profile"] = None
        snapshots = atr_record_snapshot([a, b])
        self.assertEqual([item["profile_sha256"] for item in snapshots], [None, None])

    def test_snapshot_retains_canonical_ids_and_gold_without_profiles(self):
        record = response()
        record["steps"].reverse()
        snapshot = atr_record_snapshot([record])[0]
        self.assertEqual(set(snapshot), {"response_id", "case_id", "split", "task_id", "step_ids", "gold_token_ids", "profile_sha256"})
        self.assertEqual(snapshot["step_ids"], ["r0-s0", "r0-s1"])
        self.assertEqual(snapshot["gold_token_ids"], [2, 10])
        self.assertEqual(len(snapshot["profile_sha256"]), 64)
        changed = deepcopy(record)
        changed["steps"][0]["profile"][0][0][0] += 1
        self.assertNotEqual(snapshot["profile_sha256"], atr_record_snapshot([changed])[0]["profile_sha256"])

    def test_mixed_semantic_contracts_are_rejected(self):
        first = response()
        for field in ("task_id", "tokenizer", "alignment", "feature_contract"):
            other = response("other", "test", offset=3)
            if field == "task_id":
                other[field] = "other-task"
            elif field == "tokenizer":
                other[field]["revision"] = "fixture-v2"
            elif field == "alignment":
                other[field]["special_tokens"] = "included"
            else:
                other[field]["reconstruction"]["revision"] = "fixture-v2"
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, "identical.*contracts"):
                validate_atr_records([first, other])

    def test_profile_axes_width_and_numbers_checked(self):
        for mutate in (
            lambda r: r["steps"][0].update(profile=[[[1, 2], [3, 4]]]),
            lambda r: r["steps"][0]["profile"][0].pop(),
            lambda r: r["steps"][0]["profile"][0][0].pop(),
            lambda r: r["steps"][0]["profile"][0][0].clear(),
        ):
            record = response()
            mutate(record)
            with self.assertRaises(ValidationError):
                validate_atr_records([record])
        for value in (float("nan"), float("inf"), -1.0, True, "1"):
            record = response()
            record["steps"][0]["profile"][0][0][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValidationError, "finite nonnegative"):
                validate_atr_records([record])

    def test_unique_layer_and_head_ids_and_primitive_reconstruction_parameters(self):
        for field in ("layer_ids", "kv_head_ids"):
            record = response()
            record["feature_contract"][field] = ["duplicate", "duplicate"]
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, "unique strings"):
                validate_atr_records([record])
        for value in ({"bad": float("nan")}, {"bad": object()}, []):
            record = response()
            record["feature_contract"]["reconstruction"]["parameters"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                validate_atr_records([record])

    def test_real_provenance_requires_explicit_evidence_and_revisions(self):
        record = response()
        record["source_kind"] = "reconstructed_recorded"
        record["provenance"] = {"collection_run_id": "run-1", "profile_source_sha256": "a" * 64,
                                "split_assignment_id": "split-1", "alignment_evidence_id": "alignment-1"}
        self.assertEqual(validate_atr_records([record])[0]["source_kind"], "reconstructed_recorded")
        for field in record["provenance"]:
            changed = deepcopy(record)
            changed["provenance"].pop(field)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                validate_atr_records([changed])
        for value in ("unknown", "Unassigned", " "):
            changed = deepcopy(record)
            changed["tokenizer"]["revision"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                validate_atr_records([changed])
        record["feature_contract"]["reconstruction"]["revision"] = "unknown"
        with self.assertRaises(ValidationError):
            validate_atr_records([record])

    def test_strict_fields_and_synthetic_provenance(self):
        record = response()
        record["provenance"]["synthetic"] = False
        with self.assertRaisesRegex(ValidationError, "synthetic=true"):
            validate_atr_records([record])
        for field in ("unexpected", "steps"):
            record = response()
            if field == "unexpected":
                record[field] = "bad"
            else:
                record.pop(field)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                validate_atr_records([record])

    def test_alignment_rejects_contradictory_special_token_flags(self):
        for field in ("bos_included", "eos_included"):
            record = response()
            record["alignment"][field] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, "excludes special tokens"):
                validate_atr_records([record])
            record["alignment"]["special_tokens"] = "included"
            self.assertEqual(len(validate_atr_records([record])), 1)

    def test_malformed_field_types_and_unpaired_unicode_are_validation_errors(self):
        for field, value in (("split", []), ("source_kind", {}), ("response_id", "\ud800")):
            record = response()
            record[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                validate_atr_records([record])
        record = response()
        record["alignment"]["special_tokens"] = []
        with self.assertRaises(ValidationError):
            validate_atr_records([record])

    def test_metadata_boolean_and_numeric_values_are_distinct_contracts(self):
        first, second = response(), response("other", "test", offset=3)
        first["feature_contract"]["reconstruction"]["parameters"] = {"flag": True}
        second["feature_contract"]["reconstruction"]["parameters"] = {"flag": 1}
        with self.assertRaisesRegex(ValidationError, "identical.*contracts"):
            validate_atr_records([first, second])

    def test_profile_budget_precedes_copying(self):
        with patch("artifacts.janus_artifact.atr_data.MAX_ATR_PROFILE_ELEMENTS", 19), \
             patch("artifacts.janus_artifact.atr_data.deepcopy") as copying:
            with self.assertRaisesRegex(ValidationError, "element budget"):
                validate_atr_records([response()])
            copying.assert_not_called()

    def test_load_json_array_object_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            first, second = response("a"), response("b", "test", offset=2)
            for text, count in ((json.dumps(first, indent=2), 1),
                                (json.dumps([second, first]), 2),
                                (json.dumps(second) + "\n" + json.dumps(first) + "\n", 2)):
                path.write_text(text)
                self.assertEqual(len(load_atr_records(path)), count)

    def test_json_byte_budget_duplicate_keys_and_nonfinite_literals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            path.write_text(json.dumps(response()))
            with patch("artifacts.janus_artifact.atr_data.MAX_ATR_FILE_BYTES", 20):
                with self.assertRaisesRegex(ValidationError, "byte budget"):
                    load_atr_records(path)
            path.write_text('{"response_id":"a","response_id":"b"}')
            with self.assertRaisesRegex(ValidationError, "duplicate JSON object key"):
                load_atr_records(path)
            record = response()
            record["steps"][0]["profile"][0][0][0] = float("nan")
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValidationError, "invalid JSON number"):
                load_atr_records(path)

    def test_grouped_split_reproducible_8_1_1_and_keeps_case_responses_together(self):
        records = [response(f"r{i}", offset=i) for i in range(10)]
        assigned = assign_atr_grouped_splits(records)
        self.assertEqual({split: sum(r["split"] == split for r in assigned)
                          for split in ("train", "validation", "test")},
                         {"train": 8, "validation": 1, "test": 1})
        self.assertEqual(assigned, assign_atr_grouped_splits(reversed(records)))
        records.append(response("extra", "test", case_id="case-r0", offset=99))
        assigned = assign_atr_grouped_splits(records)
        self.assertEqual(len({r["split"] for r in assigned if r["case_id"] == "case-r0"}), 1)
        self.assertTrue(all(r["split"] == "train" for r in records[:-1]))

    def test_grouped_split_records_new_assignment_evidence_without_changing_input(self):
        original = response()
        original["source_kind"] = "reconstructed_recorded"
        original["provenance"] = {"collection_run_id": "run-1", "profile_source_sha256": "a" * 64,
                                  "split_assignment_id": "external-split-1", "alignment_evidence_id": "alignment-1"}
        assigned = assign_atr_grouped_splits([original])[0]
        provenance = assigned["provenance"]
        self.assertEqual(provenance["parent_split_assignment_id"], "external-split-1")
        self.assertTrue(provenance["split_assignment_id"].startswith("atr-grouped-"))
        split_contract = provenance["generated_split_contract"]
        self.assertEqual(split_contract["group_field"], "case_id")
        self.assertEqual(split_contract["seed"], "janus-atr-v1")
        self.assertEqual(split_contract["ratios"], {"train": 0.8, "validation": 0.1, "test": 0.1})
        self.assertEqual(len(split_contract["assignments_sha256"]), 64)
        self.assertEqual(original["provenance"]["split_assignment_id"], "external-split-1")
        self.assertNotIn("generated_split_contract", original["provenance"])
        synthetic = assign_atr_grouped_splits([response()])[0]
        self.assertTrue(synthetic["provenance"]["synthetic"])
        self.assertIn("generated_split_contract", synthetic["provenance"])
        changed = assign_atr_grouped_splits([original], seed="another-seed")[0]
        self.assertNotEqual(changed["provenance"]["split_assignment_id"], provenance["split_assignment_id"])

    def test_grouped_split_rejects_invalid_ratios(self):
        for value in (float("nan"), float("inf"), True, -0.1, "0.8", 10 ** 1000):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                assign_atr_grouped_splits([response()], train_ratio=value)
        with self.assertRaisesRegex(ValidationError, "sum to 1"):
            assign_atr_grouped_splits([response()], train_ratio=0.5)


if __name__ == "__main__":
    unittest.main()
