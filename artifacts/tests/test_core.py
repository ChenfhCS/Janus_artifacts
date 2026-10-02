import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from artifacts.janus_artifact.cli import main
from artifacts.janus_artifact.dataset_manifest import (
    build_prefill_dataset_manifest,
    verify_prefill_dataset_manifest,
)
from artifacts.janus_artifact.metrics import compute_dasr, compute_pasr
from artifacts.janus_artifact.legacy_npz import (
    adapt_prefill_rank_npz,
    verify_prefill_rank_npz_record,
)
from artifacts.janus_artifact.replay import replay_record
from artifacts.janus_artifact.qai import (
    QAIConfig,
    QAIRecord,
    freeze_qai_labels,
    load_qai_records,
    prefill_rank_feature,
    run_synthetic_qai_smoke,
)
from artifacts.janus_artifact.schema import (
    SCHEMA_VERSION,
    ValidationError,
    adapt_legacy_manifest,
    load_jsonl,
    validate_trace_records,
    write_jsonl,
)
from artifacts.janus_artifact.splits import (
    assign_grouped_splits,
    freeze_vocabulary,
    validate_frozen_vocabulary,
)


def prefill(trace_id="trace-a", case_id="case-a", split="unassigned"):
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "case_id": case_id,
        "split": split,
        "phase": "prefill",
        "source_kind": "synthetic_fixture",
        "provenance": {"synthetic": True},
        "trace": {
            "granularity": "page",
            "aggregation": "cumulative",
            "observations": [3, 1, 0],
        },
        "labels": {"tokens": ["train-token"]},
    }


def decoding(trace_id="trace-d", case_id="case-d", split="unassigned"):
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "case_id": case_id,
        "response_id": f"response-{case_id}",
        "split": split,
        "phase": "decoding",
        "source_kind": "synthetic_fixture",
        "provenance": {"synthetic": True},
        "trace": {
            "granularity": "page",
            "aggregation": "stepwise",
            "observations": [[1, 0], [0, 1]],
        },
        "labels": {"tokens": ["train-token", "second-token"]},
    }


class SchemaTests(unittest.TestCase):
    def test_source_provenance_is_strict(self):
        record = prefill()
        record["source_kind"] = "real_collection"
        with self.assertRaisesRegex(ValidationError, "collection_run_id"):
            validate_trace_records([record])

    def test_duplicate_trace_id_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "duplicate trace_id"):
            validate_trace_records([prefill(), prefill()])

    def test_group_split_leakage_is_rejected(self):
        records = [prefill("a", "same", "train"), decoding("b", "same", "test")]
        with self.assertRaisesRegex(ValidationError, "group split leakage"):
            validate_trace_records(records)

    def test_legacy_pointer_adapts_without_deserialization(self):
        manifest = [{
            "legacy_path": "legacy/case.npz",
            "oid_sha256": "a" * 64,
            "size_bytes": 123,
            "phase": "prefill",
            "case_id": "legacy-case",
        }]
        record = adapt_legacy_manifest(manifest)[0]
        self.assertEqual(record["source_kind"], "refactored_legacy")
        self.assertEqual(record["payload_status"], "lfs_pointer_only")
        self.assertIsNone(record["trace"])
        with self.assertRaisesRegex(ValidationError, "metadata-only"):
            replay_record(record)


class ReplayTests(unittest.TestCase):
    def test_replay_is_derived_and_does_not_claim_reproduction(self):
        original = decoding(split="test")
        replayed = replay_record(original)
        self.assertEqual(replayed["source_kind"], "replay_derived")
        self.assertEqual(replayed["provenance"]["parent_trace_ids"], ["trace-d"])
        self.assertFalse(replayed["provenance"]["real_side_channel_reproduction"])
        self.assertEqual(replayed["trace"], original["trace"])


class LegacyNpzTests(unittest.TestCase):
    def write_valid(self, path):
        np.savez_compressed(
            path,
            attn_rank=np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5),
            top_k=np.asarray(256, dtype=np.uint16),
        )

    def test_actual_tensor_adapter_does_not_infer_labels_or_split(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "trace.npz"
            self.write_valid(payload)
            record = adapt_prefill_rank_npz(
                payload, legacy_path="legacy/case_57/trace.npz", case_id="case-57"
            )
            self.assertEqual(record["source_kind"], "refactored_legacy")
            self.assertEqual(record["split"], "unassigned")
            self.assertEqual(record["labels"], {})
            self.assertEqual(record["trace"]["data_stage"], "reconstructed_sparsity")
            self.assertEqual(record["trace"]["storage"], "external_npz")
            self.assertFalse(record["provenance"]["raw_probe_trace"])
            self.assertFalse(record["evaluation_eligibility"]["pasr"])

    def test_external_replay_verifies_actual_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "trace.npz"
            self.write_valid(payload)
            record = adapt_prefill_rank_npz(
                payload, legacy_path="legacy/trace.npz", case_id="case-57"
            )
            replayed = replay_record(record, external_payload_path=str(payload))
            self.assertTrue(replayed["provenance"]["external_payload_verified"])
            self.assertFalse(replayed["provenance"]["real_side_channel_reproduction"])
            record["trace"]["external_payload"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValidationError, "sha256 mismatch"):
                verify_prefill_rank_npz_record(record, payload)

    def test_object_array_is_rejected_without_pickle_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "unsafe.npz"
            np.savez(payload, attn_rank=np.asarray([{"unsafe": True}], dtype=object), top_k=np.asarray(1))
            with self.assertRaisesRegex(ValidationError, "pickle-backed|object dtype"):
                adapt_prefill_rank_npz(
                    payload, legacy_path="legacy/unsafe.npz", case_id="case-unsafe"
                )

    def test_unexpected_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "wrong.npz"
            np.savez(payload, unexpected=np.asarray([1], dtype=np.uint8))
            with self.assertRaisesRegex(ValidationError, "keys must equal"):
                adapt_prefill_rank_npz(
                    payload, legacy_path="legacy/wrong.npz", case_id="case-wrong"
                )


class DatasetManifestTests(unittest.TestCase):
    LEGACY_PATH = (
        "prefill_attribute_inference/legal-llama/infer_10_val/"
        "case_57/q_k_attn_rank_topk256.npz"
    )
    SOURCE = '''class NPZDataset:
    def prepare(self, one_s, npz_root):
        case_idx = int(one_s["index"])
        src_npz = os.path.join(npz_root, f"case_{case_idx}", "q_k_attn_rank_topk256.npz")
        if os.path.exists(src_npz):
            self.valid_data.append((one_s, src_npz))

class Runner:
    def __init__(self):
        self.val_csv_path = "legal_qa_100.csv"
        self.out_npz_root = "infer_10_val"
        self.out_csv_path = "infer_10_val.csv"

    def copy(self, item):
        case_idx = item["index"]
        src_npz = item["src_npz"]
        dst_case_dir = os.path.join(self.out_npz_root, f"case_{case_idx}")
        dst_npz = os.path.join(dst_case_dir, "q_k_attn_rank_topk256.npz")
        shutil.copy2(src_npz, dst_npz)
        csv_rows.append({
            "index": item["index"],
            "query": item["query"],
            "true_label": item["true_label"],
            "pred_label": item["pred_label"],
            "pred_prob": item["pred_prob"]
        })
        out_df = pd.DataFrame(csv_rows, columns=["index", "query", "true_label", "pred_label", "pred_prob"])
        out_df.to_csv(self.out_csv_path, index=False, encoding="utf-8-sig")
'''

    def prepare(self, directory, *, csv_rows=None, index=57, source=None):
        root = Path(directory)
        payload = root / "payload.npz"
        np.savez_compressed(
            payload,
            attn_rank=np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5),
            top_k=np.asarray(256, dtype=np.uint16),
        )
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        pointer = root / "pointer.npz"
        pointer.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{digest}\n"
            f"size {payload.stat().st_size}\n",
            encoding="ascii",
        )
        csv_path = root / "infer_10_val.csv"
        rows = csv_rows or [
            "index,query,true_label,pred_label,pred_prob",
            "57,Synthetic public query,Category A,Category B,0.5",
        ]
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        script = root / "legacy.py"
        script.write_text(source if source is not None else self.SOURCE, encoding="utf-8")
        legacy_path = self.LEGACY_PATH.replace("case_57", f"case_{index}")
        return csv_path, script, pointer, payload, legacy_path

    def build(self, paths):
        csv_path, script, pointer, payload, legacy_path = paths
        return build_prefill_dataset_manifest(
            dataset_id="legal-llama-infer-10-val",
            csv_path=csv_path,
            csv_legacy_path="prefill_attribute_inference/legal-llama/infer_10_val.csv",
            script_path=script,
            script_legacy_path=(
                "prefill_attribute_inference/legal-llama/"
                "load_model_infer_illness_top_10_load_npz_speed_up.py"
            ),
            pointer_path=pointer,
            payload_path=payload,
            payload_legacy_path=legacy_path,
        )

    def test_evidence_backed_join_does_not_infer_split_or_token_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.prepare(directory)
            manifest = self.build(paths)
            entry = manifest["entries"][0]
            self.assertEqual(entry["legacy_index"], 57)
            self.assertEqual(entry["case_id"], "case-57")
            self.assertEqual(entry["split"], "unassigned")
            self.assertEqual(entry["labels"]["attribute"]["value"], "Category A")
            self.assertIsNone(entry["labels"]["tokens"])
            self.assertFalse(entry["query"]["text_embedded"])
            self.assertNotIn("Synthetic public query", repr(manifest))
            self.assertTrue(entry["payload"]["pointer_matches_materialized_payload"])
            verify_prefill_dataset_manifest(
                manifest,
                csv_path=paths[0],
                script_path=paths[1],
                pointer_path=paths[2],
                payload_path=paths[3],
            )

    def test_duplicate_csv_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.prepare(
                directory,
                csv_rows=[
                    "index,query,true_label,pred_label,pred_prob",
                    "57,First,Category A,Category A,0.5",
                    "57,Second,Category B,Category B,0.6",
                ],
            )
            with self.assertRaisesRegex(ValidationError, "duplicate legacy CSV index"):
                self.build(paths)

    def test_payload_case_without_csv_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.prepare(directory, index=58)
            with self.assertRaisesRegex(ValidationError, "no row for case index 58"):
                self.build(paths)

    def test_lfs_pointer_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.prepare(directory)
            paths[2].write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{'0' * 64}\n"
                f"size {paths[3].stat().st_size}\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValidationError, "does not match"):
                self.build(paths)

    def test_missing_static_join_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.prepare(directory, source="print('not executed')\n")
            with self.assertRaisesRegex(ValidationError, "does not provide"):
                self.build(paths)


class QAIPipelineTests(unittest.TestCase):
    def test_rank_feature_is_safe_normalized_and_cross_layer_head(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "feature.npz"
            values = np.arange(4 * 4 * 3 * 8, dtype=np.uint16).reshape(4, 4, 3, 8)
            np.savez_compressed(
                payload,
                attn_rank=values,
                top_k=np.asarray(256, dtype=np.uint16),
            )
            feature, metadata = prefill_rank_feature(payload, selected_ranks=2)
            self.assertEqual(tuple(feature.shape), (8, 4, 4))
            self.assertGreaterEqual(float(feature.min()), 0.0)
            self.assertLessEqual(float(feature.max()), 1.0)
            self.assertEqual(metadata["normalization"], "per_layer_head_minmax")
            self.assertEqual(metadata["layout"], "token_channel_layer_height_head_width")
            self.assertEqual(
                metadata["axis_contract"]["histogram_reduction_axes"],
                ["query_token", "selected_rank_slot"],
            )

    def test_rank_histogram_reduces_query_axis_and_preserves_layer_head(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "axes.npz"
            ranks = np.zeros((2, 3, 2, 4), dtype=np.uint16)
            for layer in range(2):
                for head in range(3):
                    ranks[layer, head, :, (layer + head) % 4] = 9
            np.savez_compressed(
                payload,
                attn_rank=ranks,
                top_k=np.asarray(4, dtype=np.uint16),
            )
            feature, _ = prefill_rank_feature(payload, selected_ranks=1)
            self.assertEqual(tuple(feature.shape), (4, 2, 3))
            for layer in range(2):
                for head in range(3):
                    selected_key = (layer + head) % 4
                    self.assertEqual(float(feature[selected_key, layer, head]), 1.0)

    def test_token_width_padding_is_explicit_and_truncation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            short = Path(directory) / "short.npz"
            long = Path(directory) / "long.npz"
            for path, width in ((short, 6), (long, 9)):
                np.savez_compressed(
                    path,
                    attn_rank=np.arange(2 * 2 * 2 * width, dtype=np.uint16).reshape(
                        2, 2, 2, width
                    ),
                    top_k=np.asarray(width, dtype=np.uint16),
                )
            feature, metadata = prefill_rank_feature(
                short, selected_ranks=2, target_token_width=8
            )
            self.assertEqual(tuple(feature.shape), (8, 2, 2))
            self.assertTrue(torch.count_nonzero(feature[6:]).item() == 0)
            self.assertEqual(metadata["padded_token_positions"], 2)
            with self.assertRaisesRegex(ValidationError, "exceeds train-derived width"):
                prefill_rank_feature(long, selected_ranks=2, target_token_width=8)

    def test_qai_manifest_rejects_case_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "sample_id": "one",
                        "case_id": "same",
                        "split": "train",
                        "attribute": "topic",
                        "attribute_present": True,
                        "label": "a",
                        "npz_path": "one.npz",
                    },
                    {
                        "sample_id": "two",
                        "case_id": "same",
                        "split": "test",
                        "attribute": "topic",
                        "attribute_present": True,
                        "label": "a",
                        "npz_path": "two.npz",
                    },
                ],
            )
            with self.assertRaisesRegex(ValidationError, "case split leakage"):
                load_qai_records(manifest)

    def test_qai_label_vocabulary_is_train_only(self):
        records = [
            QAIRecord("a", "a", "train", "topic", True, "known-a", "/tmp/a.npz"),
            QAIRecord("b", "b", "train", "topic", True, "known-b", "/tmp/b.npz"),
            QAIRecord("c", "c", "test", "topic", True, "unseen", "/tmp/c.npz"),
        ]
        with self.assertRaisesRegex(ValidationError, "outside the frozen vocabulary"):
            freeze_qai_labels(records)

    def test_synthetic_training_gradient_checkpoint_inference_and_pasr_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_synthetic_qai_smoke(directory, device="cpu")
            self.assertTrue(report["synthetic_only"])
            self.assertFalse(report["scientific_result"])
            self.assertTrue(report["training"]["gradient_observed"])
            self.assertTrue(report["training"]["parameter_changed"])
            self.assertEqual(report["test_predictions"], 5)
            self.assertEqual(
                report["pasr_smoke_only"]["micro_denominator_present_queries"], 4
            )
            self.assertEqual(report["training"]["absent_attribute_train_records_excluded"], 1)
            self.assertEqual(report["training"]["token_width_source"], "maximum_over_train_split_only")
            self.assertEqual(report["training"]["model_input_shape"], [16, 8, 8])
            self.assertTrue(report["checkpoint_reload_verified"])
            self.assertTrue((Path(directory) / "checkpoint" / "model_state.pt").is_file())
            self.assertTrue((Path(directory) / "predictions.jsonl").is_file())
            predictions = load_jsonl(Path(directory) / "predictions.jsonl")
            absent = next(
                row for row in predictions if row["record_id"] == "synthetic-absent-test"
            )
            self.assertFalse(absent["attribute_present"])
            self.assertIsNone(absent["predicted_value"])
            self.assertEqual(absent["classification_skipped"], "attribute_absent")

    def test_qai_config_rejects_undocumented_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"architecture":"resnet18","mystery":1}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "unknown QAI config fields"):
                QAIConfig.from_json(path)


class SplitAndVocabularyTests(unittest.TestCase):
    def test_grouped_split_is_deterministic_and_isolated(self):
        records = []
        for index in range(12):
            records.append(prefill(f"p-{index}", f"case-{index}"))
            records.append(decoding(f"d-{index}", f"case-{index}"))
        first = assign_grouped_splits(records, seed="fixed")
        second = assign_grouped_splits(records, seed="fixed")
        self.assertEqual([item["split"] for item in first], [item["split"] for item in second])
        by_case = {}
        for item in first:
            by_case.setdefault(item["case_id"], set()).add(item["split"])
        self.assertTrue(all(len(splits) == 1 for splits in by_case.values()))

    def test_vocabulary_is_train_only_and_frozen(self):
        train = prefill(split="train")
        test = decoding(split="test")
        test["labels"]["tokens"] = ["test-only-token"]
        vocabulary = freeze_vocabulary([train, test])
        token_set = validate_frozen_vocabulary(vocabulary)
        self.assertIn("train-token", token_set)
        self.assertNotIn("test-only-token", token_set)
        vocabulary["tokens"].append("late-addition")
        with self.assertRaisesRegex(ValidationError, "unique and sorted|digest mismatch"):
            validate_frozen_vocabulary(vocabulary)


class MetricTests(unittest.TestCase):
    def test_pasr_denominator_excludes_absent_attribute_and_counts_missing_wrong(self):
        result = compute_pasr([
            {"record_id": "q1", "attribute": "topic", "attribute_present": True, "gold_value": "a", "predicted_value": "a"},
            {"record_id": "q2", "attribute": "topic", "attribute_present": True, "gold_value": "b", "predicted_value": None},
            {"record_id": "q3", "attribute": "topic", "attribute_present": False, "gold_value": None, "predicted_value": "a"},
        ])
        topic = result["per_attribute"]["topic"]
        self.assertEqual(topic["denominator_present_queries"], 2)
        self.assertEqual(topic["unanswered"], 1)
        self.assertEqual(topic["pasr"], 0.5)

    def test_dasr_retains_oov_and_missing_tokens_in_denominator(self):
        result = compute_dasr([
            {"response_id": "r1", "gold_tokens": ["A", "B", "OOV"], "predicted_tokens": ["A", "B", "OOV"]},
            {"response_id": "r2", "gold_tokens": ["B", "A"], "predicted_tokens": ["B"]},
        ], frozen_vocabulary={"A", "B"})
        self.assertEqual(result["micro_denominator_all_gold_tokens"], 5)
        self.assertEqual(result["micro_correct"], 3)
        self.assertEqual(result["out_of_vocabulary_gold_tokens_counted_incorrect"], 1)
        self.assertEqual(result["missing_predictions"], 1)
        self.assertAlmostEqual(result["dasr_macro"], (2 / 3 + 1 / 2) / 2)

    def test_dasr_rejects_empty_response(self):
        with self.assertRaisesRegex(ValidationError, "non-empty"):
            compute_dasr([{"response_id": "r", "gold_tokens": [], "predicted_tokens": []}])


class CliTests(unittest.TestCase):
    def test_replay_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "output.jsonl"
            source.write_text(json.dumps(prefill(split="train")) + "\n", encoding="utf-8")
            code = main(["replay", "--input", str(source), "--output", str(output)])
            self.assertEqual(code, 0)
            replayed = load_jsonl(output)
            self.assertEqual(replayed[0]["source_kind"], "replay_derived")


if __name__ == "__main__":
    unittest.main()
