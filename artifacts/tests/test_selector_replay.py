"""Standard-library-only selector replay tests; no tensors, models or GPU."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from artifacts.janus_artifact.schema import ValidationError
from artifacts.janus_artifact.selector_replay import (
    DeterministicTopKSelector, SelectorReplayConfig, build_selector_manifest,
    validate_selector_manifest,
)


def config(**changes):
    values = {
        "run_id": "a" * 32, "model_id": "synthetic-model", "model_revision": "fixture-v1",
        "tokenizer_id": "synthetic-tokenizer", "tokenizer_revision": "fixture-v1",
        "model_config_sha256": "b" * 64, "weights_sha256": None,
        "source_kind": "synthetic_tensor_fixture", "seed": 19,
        "prompt_token_ids": [7, 8, 9], "teacher_forced_token_ids": [10, 11],
        "layers": 2, "query_heads": 4, "kv_heads": 2, "head_dim": 3, "top_k": 1,
    }
    values.update(changes)
    return SelectorReplayConfig(**values)


def rows(cfg):
    return [{"step_index": step, "layer_index": layer, "query_head_index": head,
             "scores": [float((position * 3 + head + layer + step) % 5)
                        for position in range(len(cfg.prompt_token_ids) + step)]}
            for step in range(len(cfg.teacher_forced_token_ids))
            for layer in range(cfg.layers) for head in range(cfg.query_heads)]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def rehash(manifest):
    manifest["manifest_sha256"] = digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    return manifest


class SelectorReplayTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.rows = rows(self.cfg)
        self.manifest = build_selector_manifest(self.cfg, self.rows)

    def test_config_frozen_token_snapshot_and_static_cache_capacity(self):
        prompt, forced = [1, 2, 3], [4, 5]
        cfg = config(prompt_token_ids=prompt, teacher_forced_token_ids=forced)
        prompt.append(99)
        forced[0] = 99
        self.assertEqual(cfg.prompt_token_ids, (1, 2, 3))
        self.assertEqual(cfg.teacher_forced_token_ids, (4, 5))
        self.assertEqual(cfg.cache_capacity, 4)
        self.assertEqual(cfg.to_dict()["prompt_token_ids"], [1, 2, 3])

    def test_topk_score_order_and_absolute_position_ties_are_deterministic(self):
        selector = DeterministicTopKSelector()
        self.assertEqual(selector.select([3.0, 9.0, 9.0], [2, 7, 4], 1, 0), (4,))
        self.assertEqual(selector.select([3.0, 9.0, 9.0], [2, 7, 4], 2, 999), (4, 7))
        self.assertEqual(selector.select([3.0, 9.0, 9.0], [2, 7, 4], 100, 19), (2, 4, 7))

    def test_manifest_query_target_alignment_gqa_and_opaque_ids(self):
        self.assertEqual(set(self.manifest), {"schema_version", "config", "scores_sha256", "sequence_sha256", "steps", "manifest_sha256"})
        for index, step in enumerate(self.manifest["steps"]):
            self.assertEqual(step["cache_length"], 3 + index)
            self.assertEqual(step["query_position"], 2 + index)
            self.assertEqual(step["target_token_id"], 10 + index)
            self.assertEqual(step["step_id"], hashlib.sha256(f"{'a' * 32}|step|{index}".encode()).hexdigest()[:32])
            self.assertEqual([selection["kv_head_index"] for selection in step["selections"]], [0, 0, 1, 1] * 2)
            self.assertTrue(all(step["query_position"] in selection["absolute_kv_positions"] for selection in step["selections"]))
        self.assertEqual(self.manifest["steps"][0]["selections"][0]["absolute_kv_positions"], [1, 2])
        self.assertEqual(self.manifest["steps"][1]["selections"][0]["absolute_kv_positions"], [1, 3])

    def test_input_row_reordering_is_canonical_and_scores_digest_matches_source(self):
        self.assertEqual(self.manifest, build_selector_manifest(self.cfg, reversed(self.rows)))
        self.assertEqual(self.manifest["scores_sha256"], digest(self.rows))
        self.assertEqual(self.manifest["manifest_sha256"], digest({key: value for key, value in self.manifest.items() if key != "manifest_sha256"}))
        self.assertEqual(self.rows, rows(self.cfg))

    def test_no_latest_and_k_larger_than_cache(self):
        cfg = replace(self.cfg, include_latest=False)
        manifest = build_selector_manifest(cfg, rows(cfg))
        self.assertEqual(manifest["steps"][0]["selections"][0]["absolute_kv_positions"], [1])
        cfg = replace(self.cfg, top_k=100)
        manifest = build_selector_manifest(cfg, rows(cfg))
        for step in manifest["steps"]:
            self.assertEqual(step["selections"][0]["absolute_kv_positions"], list(range(step["cache_length"])))

    def test_custom_strategy_contract_and_immutable_score_snapshot(self):
        class First:
            name = "first"
            version = "fixture-v1"
            def select(self, scores, absolute_positions, top_k, seed):
                self.score_is_tuple = type(scores) is tuple
                return tuple(absolute_positions[:top_k])
        selector = First()
        cfg = replace(self.cfg, selector_name=selector.name, selector_version=selector.version)
        manifest = build_selector_manifest(cfg, rows(cfg), selector)
        self.assertTrue(selector.score_is_tuple)
        self.assertEqual(manifest["scores_sha256"], digest(rows(cfg)))
        self.assertEqual(manifest["steps"][0]["selections"][0]["absolute_kv_positions"], [0, 2])
        self.assertEqual(validate_selector_manifest(manifest, expected_config=cfg), manifest)
        class Mutates(First):
            def select(self, scores, absolute_positions, top_k, seed):
                scores[0] = 99
                return (0,)
        with self.assertRaisesRegex(ValidationError, "selector execution failed"):
            build_selector_manifest(cfg, rows(cfg), Mutates())

    def test_custom_strategy_name_version_count_positions_and_method_rejected(self):
        class Custom:
            name = "custom"
            version = "v1"
            def select(self, scores, absolute_positions, top_k, seed):
                return self.output
        cfg = replace(self.cfg, selector_name="custom", selector_version="v1", top_k=2)
        for output in ([0, 0], [1, 0], [True, 1], [-1, 1], [0, 3], [0], [0, 1, 2], "01", None):
            strategy = Custom()
            strategy.output = output
            with self.subTest(output=output), self.assertRaises(ValidationError):
                build_selector_manifest(cfg, rows(cfg), strategy)
        strategy = Custom()
        strategy.output = [0, 1]
        with self.assertRaisesRegex(ValidationError, "name/version"):
            build_selector_manifest(self.cfg, self.rows, strategy)
        with self.assertRaisesRegex(ValidationError, "name/version"):
            build_selector_manifest(cfg, rows(cfg))
        strategy.select = None
        with self.assertRaisesRegex(ValidationError, "callable"):
            build_selector_manifest(cfg, rows(cfg), strategy)

    def test_config_sources_digest_ids_and_text_are_strict(self):
        for name, value in (("run_id", "bad"), ("run_id", "A" * 32), ("model_config_sha256", "z" * 64),
                            ("weights_sha256", False), ("model_revision", ""), ("tokenizer_revision", " "),
                            ("source_kind", "physical"), ("selector_name", "\ud800")):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                config(**{name: value}).validate()
        cfg = config(source_kind="offline_attention_scores")
        with self.assertRaisesRegex(ValidationError, "weights_sha256"):
            cfg.validate()
        config(source_kind="offline_attention_scores", weights_sha256="c" * 64).validate()

    def test_integer_boolean_layout_and_sequence_bounds(self):
        for name, value in (("seed", -1), ("seed", True), ("query_heads", True), ("kv_heads", 3),
                            ("layers", 0), ("head_dim", 0), ("top_k", False), ("include_latest", 1),
                            ("prompt_token_ids", []), ("teacher_forced_token_ids", [True]),
                            ("prompt_token_ids", [0, -1]), ("prompt_token_ids", "123")):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                config(**{name: value}).validate()
        for name, value in (("prompt_token_ids", [0] * 257), ("teacher_forced_token_ids", [0] * 33),
                            ("layers", 33), ("query_heads", 129), ("kv_heads", 129), ("head_dim", 257)):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                config(**{name: value}).validate()
        with self.assertRaisesRegex(ValidationError, "score element budget"):
            config(prompt_token_ids=[0] * 256, teacher_forced_token_ids=[0] * 32, layers=32, query_heads=128, kv_heads=8).validate()

    def test_config_json_all_fields_duplicate_keys_finite_and_byte_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(self.cfg.to_dict()))
            self.assertEqual(SelectorReplayConfig.from_json(path), self.cfg)
            for name in ("seed", "include_latest", "selector_name", "selector_version"):
                value = self.cfg.to_dict()
                value.pop(name)
                with self.subTest(name=name), self.assertRaises(ValidationError):
                    SelectorReplayConfig.from_dict(value)
            value = self.cfg.to_dict()
            value["unknown"] = 1
            with self.assertRaises(ValidationError):
                SelectorReplayConfig.from_dict(value)
            path.write_text('{"seed":19,"seed":20}')
            with self.assertRaisesRegex(ValidationError, "duplicate"):
                SelectorReplayConfig.from_json(path)
            path.write_text('{"seed":NaN}')
            with self.assertRaisesRegex(ValidationError, "nonfinite"):
                SelectorReplayConfig.from_json(path)
            path.write_text(json.dumps(self.cfg.to_dict()))
            with patch("artifacts.janus_artifact.selector_replay.MAX_CONFIG_FILE_BYTES", 16):
                with self.assertRaisesRegex(ValidationError, "byte budget"):
                    SelectorReplayConfig.from_json(path)

    def test_rows_missing_duplicate_extra_fields_and_out_of_range_keys(self):
        with self.assertRaisesRegex(ValidationError, "missing"):
            build_selector_manifest(self.cfg, self.rows[:-1])
        changed = deepcopy(self.rows)
        changed[-1] = deepcopy(changed[0])
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            build_selector_manifest(self.cfg, changed)
        with self.assertRaisesRegex(ValidationError, "too many"):
            build_selector_manifest(self.cfg, self.rows + [self.rows[0]])
        for field, value in (("step_index", True), ("step_index", 2), ("layer_index", 2), ("query_head_index", -1)):
            changed = deepcopy(self.rows)
            changed[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                build_selector_manifest(self.cfg, changed)
        changed = deepcopy(self.rows)
        changed[0]["unknown"] = True
        with self.assertRaises(ValidationError):
            build_selector_manifest(self.cfg, changed)

    def test_score_rows_require_exact_causal_length_and_finite_builtin_numbers(self):
        class FloatSubclass(float):
            pass
        for scores in ([0, 1], [0, 1, 2, 3], (0, 1, 2), [0, True, 2], [0, float("nan"), 2],
                       [0, float("inf"), 2], [0, FloatSubclass(1), 2], [0, 10 ** 1000, 2]):
            changed = deepcopy(self.rows)
            changed[0]["scores"] = scores
            with self.subTest(scores=scores), self.assertRaises(ValidationError):
                build_selector_manifest(self.cfg, changed)
        with self.assertRaises(ValidationError):
            build_selector_manifest(self.cfg, None)

    def test_manifest_positions_reject_future_duplicate_reordered_empty_boolean(self):
        for positions in ([1, 3], [1, 1, 2], [2, 1], [], [True, 2], [-1, 2]):
            manifest = deepcopy(self.manifest)
            manifest["steps"][0]["selections"][0]["absolute_kv_positions"] = positions
            with self.subTest(positions=positions), self.assertRaises(ValidationError):
                validate_selector_manifest(rehash(manifest), expected_config=self.cfg)

    def test_manifest_latest_count_and_nonlatest_counts_rejected(self):
        for positions in ([0], [0, 1, 2]):
            manifest = deepcopy(self.manifest)
            manifest["steps"][0]["selections"][0]["absolute_kv_positions"] = positions
            with self.subTest(positions=positions), self.assertRaisesRegex(ValidationError, "include_latest/count"):
                validate_selector_manifest(rehash(manifest))
        cfg = replace(self.cfg, include_latest=False)
        manifest = build_selector_manifest(cfg, rows(cfg))
        manifest["steps"][0]["selections"][0]["absolute_kv_positions"] = [0, 1]
        with self.assertRaisesRegex(ValidationError, "count"):
            validate_selector_manifest(rehash(manifest))

    def test_manifest_missing_reordered_steps_heads_gqa_and_target_alignment(self):
        changes = (
            lambda m: m["steps"].reverse(), lambda m: m["steps"].pop(),
            lambda m: m["steps"][0]["selections"].reverse(),
            lambda m: m["steps"][0]["selections"].pop(),
            lambda m: m["steps"][0]["selections"][0].update(kv_head_index=1),
            lambda m: m["steps"][0].update(target_token_id=99),
            lambda m: m["steps"][0].update(query_position=3),
            lambda m: m["steps"][0].update(cache_length=4),
            lambda m: m["steps"][0].update(step_id="b" * 32),
            lambda m: m["steps"][0].update(step_index=True),
        )
        for change in changes:
            manifest = deepcopy(self.manifest)
            change(manifest)
            with self.assertRaises(ValidationError):
                validate_selector_manifest(rehash(manifest))

    def test_expected_config_binds_each_provenance_layout_sequence_and_selector_field(self):
        changes = {
            "run_id": "d" * 32, "model_id": "other", "model_revision": "v2", "tokenizer_id": "other",
            "tokenizer_revision": "v2", "model_config_sha256": "d" * 64, "weights_sha256": "d" * 64,
            "source_kind": "offline_attention_scores", "seed": 20, "prompt_token_ids": [1, 2, 3],
            "teacher_forced_token_ids": [12, 11], "layers": 1, "query_heads": 2, "kv_heads": 1,
            "head_dim": 4, "top_k": 2, "include_latest": False, "selector_name": "other", "selector_version": "v2",
        }
        for name, value in changes.items():
            values = {name: value}
            if name == "source_kind":
                values["weights_sha256"] = "d" * 64
            expected = replace(self.cfg, **values)
            with self.subTest(name=name), self.assertRaisesRegex(ValidationError, "expected_config mismatch"):
                validate_selector_manifest(self.manifest, expected_config=expected)
        self.assertEqual(validate_selector_manifest(self.manifest, self.cfg.to_dict()), self.manifest)

    def test_manifest_digests_strict_unknown_fields_and_copy_independence(self):
        for field in ("scores_sha256", "sequence_sha256", "manifest_sha256"):
            manifest = deepcopy(self.manifest)
            manifest[field] = "bad"
            with self.subTest(field=field), self.assertRaises(ValidationError):
                validate_selector_manifest(manifest)
        manifest = deepcopy(self.manifest)
        manifest["steps"][0]["selections"][0]["absolute_kv_positions"] = [0, 2]
        with self.assertRaisesRegex(ValidationError, "manifest digest"):
            validate_selector_manifest(manifest)
        manifest = deepcopy(self.manifest)
        manifest["sequence_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValidationError, "sequence digest"):
            validate_selector_manifest(rehash(manifest))
        manifest = deepcopy(self.manifest)
        manifest["unknown"] = 1
        with self.assertRaises(ValidationError):
            validate_selector_manifest(manifest)
        copied = validate_selector_manifest(self.manifest)
        copied["steps"][0]["selections"][0]["absolute_kv_positions"].append(0)
        self.assertEqual(self.manifest["steps"][0]["selections"][0]["absolute_kv_positions"], [1, 2])

    def test_builtin_string_fields_keys_and_custom_metadata_are_strict(self):
        class StringSubclass(str):
            pass
        cfg = self.cfg.to_dict()
        cfg[StringSubclass("run_id")] = cfg.pop("run_id")
        with self.assertRaises(ValidationError):
            SelectorReplayConfig.from_dict(cfg)
        with self.assertRaises(ValidationError):
            replace(self.cfg, source_kind=StringSubclass("synthetic_tensor_fixture")).validate()
        for field in ("schema_version", "step_id"):
            manifest = deepcopy(self.manifest)
            target = manifest if field == "schema_version" else manifest["steps"][0]
            target[field] = StringSubclass(target[field])
            with self.subTest(field=field), self.assertRaises(ValidationError):
                validate_selector_manifest(rehash(manifest))
        class Strategy:
            name = StringSubclass("score_topk")
            version = "1"
            def select(self, scores, positions, top_k, seed):
                return (0,)
        with self.assertRaisesRegex(ValidationError, "name/version"):
            build_selector_manifest(self.cfg, self.rows, Strategy())

    def test_self_consistent_digest_does_not_certify_topk_ranking(self):
        # A manifest alone lacks score rows: changing legal membership and rehashing
        # remains structurally valid. This limitation is explicit, not authentication.
        manifest = deepcopy(self.manifest)
        manifest["steps"][0]["selections"][0]["absolute_kv_positions"] = [0, 2]
        self.assertEqual(validate_selector_manifest(rehash(manifest), self.cfg)["steps"][0]["selections"][0]["absolute_kv_positions"], [0, 2])


if __name__ == "__main__":
    unittest.main()
