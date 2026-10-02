"""CPU mathematical/static-storage regressions; no models, GPU or probe calls."""
from copy import deepcopy
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import torch

from artifacts.janus_artifact.schema import ValidationError
from artifacts.janus_artifact.selector_replay import SelectorReplayConfig, build_selector_manifest, _sha256
from artifacts.janus_artifact.sparse_attention import StaticKVCache, selected_attention, MAX_CACHE_BYTES


def config(**changes):
    defaults = dict(run_id="1" * 32, model_id="synthetic-attention-core", model_revision="fixture-v1",
        tokenizer_id="synthetic-tokens", tokenizer_revision="fixture-v1",
        model_config_sha256="a" * 64, weights_sha256=None, source_kind="synthetic_tensor_fixture",
        seed=19, prompt_token_ids=[3, 5, 7, 9], teacher_forced_token_ids=[11, 13, 17],
        layers=2, query_heads=4, kv_heads=2, head_dim=4, top_k=1, include_latest=True)
    defaults.update(changes)
    result = SelectorReplayConfig(**defaults)
    result.validate()
    return result


def manifest_for(cfg):
    rows = []
    for step in range(len(cfg.teacher_forced_token_ids)):
        length = len(cfg.prompt_token_ids) + step
        for layer in range(cfg.layers):
            for head in range(cfg.query_heads):
                scores = [float(-position) for position in range(length)]
                scores[head % min(3, length)] = 10.0 + layer
                rows.append(dict(step_index=step, layer_index=layer, query_head_index=head, scores=scores))
    return build_selector_manifest(cfg, rows)


def bind_digest(manifest):
    manifest["manifest_sha256"] = _sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    return manifest


def row_pair(cfg, layer, position, dtype=torch.float32):
    base = torch.arange(cfg.kv_heads * cfg.head_dim, dtype=dtype).reshape(cfg.kv_heads, cfg.head_dim)
    return (base + 3 * position + 11 * layer) / 8, (base + 5 * position - 7 * layer) / 4


def fill(cache, length=None):
    length = len(cache.config.prompt_token_ids) if length is None else length
    for layer in range(cache.config.layers):
        for position in range(length):
            key, value = row_pair(cache.config, layer, position, cache.dtype)
            cache.write(layer, position, key, value)


def query_for(cfg, dtype=torch.float32):
    generator = torch.Generator(device="cpu").manual_seed(29)
    return torch.randn(cfg.query_heads, cfg.head_dim, dtype=dtype, generator=generator) / 4


def dense_selected_reference(query, cache, manifest, step, layer):
    """Dense causal scores masked to exactly the manifest set, for finite fixtures."""
    length = len(cache.config.prompt_token_ids) + step
    expected = []
    ratio = cache.config.query_heads // cache.config.kv_heads
    entries = [entry for entry in manifest["steps"][step]["selections"] if entry["layer_index"] == layer]
    for head, entry in enumerate(entries):
        keys = cache.keys[layer, head // ratio, :length]
        values = cache.values[layer, head // ratio, :length]
        scores = torch.matmul(keys, query[head]) / math.sqrt(cache.config.head_dim)
        mask = torch.zeros(length, dtype=torch.bool)
        mask[entry["absolute_kv_positions"]] = True
        probabilities = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=0)
        expected.append(torch.matmul(probabilities, values))
    return torch.stack(expected)


class SparseAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = config()
        self.manifest = manifest_for(self.config)
        self.cache = StaticKVCache(self.config)
        fill(self.cache)
        self.query = query_for(self.config)

    def test_static_shape_capacity_and_default_cpu_dtype(self):
        self.assertEqual(self.cache.capacity, 6)
        self.assertEqual(tuple(self.cache.keys.shape), (2, 2, 6, 4))
        self.assertEqual(self.cache.keys.device.type, "cpu")
        self.assertEqual(self.cache.keys.dtype, torch.float32)
        self.assertEqual(self.cache.valid_lengths, (4, 4))
        self.assertEqual(self.cache.cache_bytes, 2 * 2 * 2 * 6 * 4 * 4)
        with self.assertRaises(TypeError):
            self.cache.valid_lengths[0] = 0

    def test_static_storage_pointer_stays_stable_across_all_appends(self):
        cache = StaticKVCache(self.config)
        pointers = (cache.keys.data_ptr(), cache.values.data_ptr())
        for layer in range(self.config.layers):
            for position in range(cache.capacity):
                key, value = row_pair(self.config, layer, position)
                cache.write(layer, position, key, value)
                self.assertEqual((cache.keys.data_ptr(), cache.values.data_ptr()), pointers)
                torch.testing.assert_close(cache.keys[layer, :, position], key)
                torch.testing.assert_close(cache.values[layer, :, position], value)
        self.assertEqual(cache.valid_lengths, (6, 6))

    def test_budget_checked_before_allocation_independently_of_score_budget(self):
        cfg = config(layers=32, query_heads=128, kv_heads=128, head_dim=256,
                     prompt_token_ids=list(range(256)), teacher_forced_token_ids=[11])
        cfg.validate()  # score budget is legitimate, but the KV allocation is much larger.
        with patch("torch.empty", side_effect=AssertionError("must not allocate")) as allocate:
            with self.assertRaisesRegex(ValidationError, "64 MiB"):
                StaticKVCache(cfg)
        allocate.assert_not_called()
        self.assertGreater(2 * 32 * 128 * 256 * 256 * 4, MAX_CACHE_BYTES)

    def test_invalid_config_dtype_and_device_fail_before_allocation(self):
        for dtype in (torch.int64, torch.float16, torch.bfloat16, "float32"):
            with self.subTest(dtype=dtype), patch("torch.empty") as allocate, self.assertRaises(ValidationError):
                StaticKVCache(self.config, dtype=dtype)
            allocate.assert_not_called()
        with patch("torch.empty") as allocate, self.assertRaises(ValidationError):
            StaticKVCache(self.config, device="meta")
        allocate.assert_not_called()
        with patch("torch.empty") as allocate, self.assertRaises(ValidationError):
            StaticKVCache(replace(self.config, query_heads=3))
        allocate.assert_not_called()

    def test_write_rejects_overwrite_gap_indices_and_full_capacity(self):
        key, value = row_pair(self.config, 0, 4)
        for layer, position in ((True, 4), (-1, 4), (2, 4), (0, True), (0, 4.0), (0, -1),
                                (0, 3), (0, 5), (0, 6)):
            with self.subTest(layer=layer, position=position), self.assertRaises(ValidationError):
                self.cache.write(layer, position, key, value)
        self.assertEqual(self.cache.valid_lengths, (4, 4))
        self.cache.write(0, 4, key, value)
        key, value = row_pair(self.config, 0, 5)
        self.cache.write(0, 5, key, value)
        with self.assertRaises(ValidationError):
            self.cache.write(0, 6, key, value)

    def test_write_validates_both_rows_before_mutating(self):
        self.cache.keys[0, :, 4].fill_(41)
        self.cache.values[0, :, 4].fill_(42)
        key, value = row_pair(self.config, 0, 4)
        cases = ((key[:, :-1], value), (key.to(torch.float64), value),
                 (key, value[:, :-1]), (key, value.to(torch.float64)),
                 (torch.full_like(key, torch.nan), value), (key, torch.full_like(value, torch.inf)),
                 (torch.empty(key.shape, device="meta"), value), (key, [[1.0]]))
        for incoming_key, incoming_value in cases:
            with self.subTest(key_type=type(incoming_key), value_type=type(incoming_value)), self.assertRaises(ValidationError):
                self.cache.write(0, 4, incoming_key, incoming_value)
            self.assertTrue(bool((self.cache.keys[0, :, 4] == 41).all()))
            self.assertTrue(bool((self.cache.values[0, :, 4] == 42).all()))
            self.assertEqual(self.cache.valid_lengths, (4, 4))

    def test_cache_write_avoids_retaining_input_autograd_graph(self):
        key, value = row_pair(self.config, 0, 4)
        key.requires_grad_()
        value.requires_grad_()
        self.cache.write(0, 4, key, value)
        self.assertFalse(self.cache.keys.requires_grad)
        self.assertFalse(self.cache.values.requires_grad)
        self.assertIsNone(self.cache.keys.grad_fn)

    def test_gather_only_accepts_nonempty_sorted_unique_causal_builtin_positions(self):
        for positions in ([], [2, 1], [1, 1], [-1], [4], [True], [1.0], torch.tensor([1]), "1"):
            with self.subTest(positions=positions), self.assertRaises(ValidationError):
                self.cache.gather(0, 0, positions)
        for layer, head in ((-1, 0), (2, 0), (False, 0), (0, -1), (0, 2), (0, True)):
            with self.subTest(layer=layer, head=head), self.assertRaises(ValidationError):
                self.cache.gather(layer, head, [0])
        keys, values = self.cache.gather(0, 1, [0, 2])
        torch.testing.assert_close(keys, self.cache.keys[0, 1, [0, 2]])
        torch.testing.assert_close(values, self.cache.values[0, 1, [0, 2]])

    def test_dense_masked_reference_matches_selected_float32_and_float64_all_layers_steps(self):
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-12)):
            cache = StaticKVCache(self.config, dtype=dtype)
            fill(cache)
            query = query_for(self.config, dtype)
            for step in range(len(self.config.teacher_forced_token_ids)):
                if step:
                    for layer in range(self.config.layers):
                        position = len(self.config.prompt_token_ids) + step - 1
                        cache.write(layer, position, *row_pair(self.config, layer, position, dtype))
                for layer in range(self.config.layers):
                    with self.subTest(dtype=dtype, step=step, layer=layer):
                        actual, audit = selected_attention(query, cache, self.manifest, step_index=step, layer_index=layer)
                        expected = dense_selected_reference(query, cache, self.manifest, step, layer)
                        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                        self.assertEqual(audit["query_position"], 4 + step - 1)
                        self.assertEqual(audit["cache_valid_length"], 4 + step)
                        self.assertEqual(actual.dtype, dtype)
                        self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_gqa_subsets_and_union_are_distinct_and_audit_is_logical_only(self):
        _, audit = selected_attention(self.query, self.cache, self.manifest, step_index=0, layer_index=0)
        group0, group1 = audit["kv_head_groups"]
        self.assertEqual(group0["union_selected_positions"], [0, 1, 3])
        self.assertEqual(group0["query_heads"][0]["absolute_kv_positions"], [0, 3])
        self.assertEqual(group0["query_heads"][1]["absolute_kv_positions"], [1, 3])
        self.assertEqual(group1["union_selected_positions"], [0, 2, 3])
        self.assertEqual([head["query_head_index"] for head in group1["query_heads"]], [2, 3])
        self.assertFalse(audit["physical_cacheline_access_verified"])
        self.assertFalse(audit["physical_gpu_memory_traffic_verified"])
        self.assertFalse(audit["is_probe_observation"])
        self.assertEqual(audit["gathered_bytes_theoretical"], 2 * 2 * 3 * 4 * 4)
        self.assertEqual(audit["source_cache_bytes_theoretical"], 2 * 2 * 6 * 4 * 4)
        self.assertEqual(audit["source_valid_prefix_bytes_theoretical"], 2 * 2 * 4 * 4 * 4)

    def test_unselected_written_and_future_nan_cannot_change_selected_output(self):
        baseline, _ = selected_attention(self.query, self.cache, self.manifest, step_index=0, layer_index=0)
        # KV0 pos2 and KV1 pos1 belong to none of that group's selected subsets.
        self.cache.keys[0, 0, 2].fill_(torch.nan)
        self.cache.values[0, 0, 2].fill_(1e30)
        self.cache.keys[0, 1, 1].fill_(torch.inf)
        self.cache.values[0, 1, 1].fill_(torch.nan)
        self.cache.keys[:, :, 4:].fill_(torch.nan)
        self.cache.values[:, :, 4:].fill_(torch.inf)
        actual, _ = selected_attention(self.query, self.cache, self.manifest, step_index=0, layer_index=0)
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)

    def test_selected_nan_or_infinite_values_fail(self):
        for target, value in (("keys", torch.nan), ("values", torch.inf)):
            cache = StaticKVCache(self.config)
            fill(cache)
            getattr(cache, target)[0, 0, 0].fill_(value)
            with self.subTest(target=target), self.assertRaises(ValidationError):
                selected_attention(self.query, cache, self.manifest, step_index=0, layer_index=0)

    def test_query_shape_dtype_device_nonfinite_and_non_tensor_rejected(self):
        invalid = (self.query[:, :-1], self.query.to(torch.float64), torch.full_like(self.query, torch.nan),
                   torch.empty(self.query.shape, device="meta"), [[1.0]])
        for query in invalid:
            with self.subTest(query_type=type(query)), self.assertRaises(ValidationError):
                selected_attention(query, self.cache, self.manifest, step_index=0, layer_index=0)

    def test_finite_inputs_that_overflow_selected_logits_fail_closed(self):
        self.cache.keys[0, 0, 0].fill_(1e30)
        query = torch.full_like(self.query, 1e30)
        self.assertTrue(bool(torch.isfinite(query).all()))
        with self.assertRaisesRegex(ValidationError, "logits"):
            selected_attention(query, self.cache, self.manifest, step_index=0, layer_index=0)

    def test_softmax_and_output_are_independently_checked_for_nonfinite(self):
        with patch("torch.softmax", return_value=torch.tensor([torch.nan, 0.0])), self.assertRaisesRegex(ValidationError, "probabilities"):
            selected_attention(self.query, self.cache, self.manifest, step_index=0, layer_index=0)
        matmul = torch.matmul
        def bad_value_product(first, second):
            result = matmul(first, second)
            if first.ndim == 1 and second.ndim == 2:
                return torch.full_like(result, torch.inf)
            return result
        with patch("torch.matmul", side_effect=bad_value_product), self.assertRaisesRegex(ValidationError, "output"):
            selected_attention(self.query, self.cache, self.manifest, step_index=0, layer_index=0)

    def test_cache_missing_or_future_rows_fail_exact_length_contract(self):
        for length in (3, 5):
            cache = StaticKVCache(self.config)
            fill(cache, length)
            with self.subTest(length=length), self.assertRaisesRegex(ValidationError, "valid length"):
                selected_attention(self.query, cache, self.manifest, step_index=0, layer_index=0)
        with self.assertRaisesRegex(ValidationError, "valid length"):
            selected_attention(self.query, self.cache, self.manifest, step_index=1, layer_index=0)

    def test_invalid_step_layer_and_cache_metadata_fail(self):
        for step, layer in ((True, 0), (-1, 0), (3, 0), (0.0, 0), (0, True), (0, 2)):
            with self.subTest(step=step, layer=layer), self.assertRaises(ValidationError):
                selected_attention(self.query, self.cache, self.manifest, step_index=step, layer_index=layer)
        with self.assertRaises(ValidationError):
            selected_attention(self.query, object(), self.manifest, step_index=0, layer_index=0)
        self.cache.capacity += 1
        with self.assertRaisesRegex(ValidationError, "capacity"):
            self.cache.gather(0, 0, [0])

    def test_attention_requires_manifest_and_explicit_config_to_match_cache_identity(self):
        changes = ({"model_id": "other"}, {"model_revision": "other"}, {"tokenizer_revision": "other"},
                   {"model_config_sha256": "b" * 64}, {"weights_sha256": "c" * 64},
                   {"seed": 20}, {"prompt_token_ids": [3, 5, 7, 10]},
                   {"teacher_forced_token_ids": [11, 13, 19]}, {"include_latest": False})
        for change in changes:
            other = replace(self.config, **change)
            other_manifest = manifest_for(other)
            with self.subTest(change=change), self.assertRaisesRegex(ValidationError, "expected_config"):
                selected_attention(self.query, self.cache, other_manifest, step_index=0, layer_index=0)
            with self.assertRaisesRegex(ValidationError, "expected_config"):
                selected_attention(self.query, self.cache, self.manifest, step_index=0, layer_index=0, expected_config=other)

    def test_semantically_invalid_manifest_fails_even_with_rebound_digest(self):
        for positions in ([], [1, 0], [0, 0], [4], [-1, 3], [True, 3]):
            malformed = deepcopy(self.manifest)
            malformed["steps"][0]["selections"][0]["absolute_kv_positions"] = positions
            bind_digest(malformed)
            with self.subTest(positions=positions), self.assertRaises(ValidationError):
                selected_attention(self.query, self.cache, malformed, step_index=0, layer_index=0)
        malformed = deepcopy(self.manifest)
        malformed["steps"][0]["selections"][0]["kv_head_index"] = 1
        with self.assertRaises(ValidationError):
            selected_attention(self.query, self.cache, bind_digest(malformed), step_index=0, layer_index=0)
        malformed = deepcopy(self.manifest)
        malformed["steps"][0]["selections"].reverse()
        with self.assertRaises(ValidationError):
            selected_attention(self.query, self.cache, bind_digest(malformed), step_index=0, layer_index=0)
        malformed = deepcopy(self.manifest)
        malformed["steps"][0]["query_position"] = 4
        with self.assertRaises(ValidationError):
            selected_attention(self.query, self.cache, bind_digest(malformed), step_index=0, layer_index=0)

    def test_source_index_select_is_once_per_kv_group_and_finite_scans_use_only_small_buffers(self):
        calls, finite_sources = [], []
        index_select, isfinite = torch.index_select, torch.isfinite
        cache_storage = {self.cache.keys.untyped_storage().data_ptr(), self.cache.values.untyped_storage().data_ptr()}
        def record_select(source, dimension, indices):
            calls.append((source.data_ptr(), tuple(source.shape), dimension, indices.tolist()))
            return index_select(source, dimension, indices)
        def record_finite(source):
            finite_sources.append((source.untyped_storage().data_ptr(), tuple(source.shape)))
            return isfinite(source)
        with patch("torch.index_select", side_effect=record_select), patch("torch.isfinite", side_effect=record_finite):
            selected_attention(self.query, self.cache, self.manifest, step_index=0, layer_index=0)
        source_pointers = {self.cache.keys[0, head].data_ptr() for head in range(2)} | {
            self.cache.values[0, head].data_ptr() for head in range(2)}
        source_calls = [entry for entry in calls if entry[0] in source_pointers]
        self.assertEqual(len(source_calls), 4)
        self.assertEqual(sorted(entry[3] for entry in source_calls), [[0, 1, 3], [0, 1, 3], [0, 2, 3], [0, 2, 3]])
        self.assertTrue(all(entry[1] == (6, 4) and entry[2] == 0 for entry in source_calls))
        self.assertEqual(len(calls), 4 + 2 * self.config.query_heads)
        self.assertTrue(all(pointer not in cache_storage for pointer, _ in finite_sources))

    def test_cache_operations_do_not_cat_clone_contiguous_or_expand_full_cache(self):
        forbidden = AssertionError("full cache copy/expansion forbidden")
        key, value = row_pair(self.config, 0, 4)
        with patch("torch.cat", side_effect=forbidden), patch.object(torch.Tensor, "clone", side_effect=forbidden), \
                patch.object(torch.Tensor, "contiguous", side_effect=forbidden), \
                patch.object(torch.Tensor, "repeat_interleave", side_effect=forbidden):
            selected_attention(self.query, self.cache, self.manifest, step_index=0, layer_index=0)
            self.cache.gather(0, 0, [0, 3])
            self.cache.write(0, 4, key, value)
        self.assertEqual(self.cache.valid_lengths[0], 5)

    def test_single_token_prompt_last_step_capacity_has_no_off_by_one(self):
        cfg = config(prompt_token_ids=[3], teacher_forced_token_ids=[7, 11], layers=1,
                     query_heads=2, kv_heads=1, top_k=1)
        manifest = manifest_for(cfg)
        cache = StaticKVCache(cfg)
        fill(cache)
        query = query_for(cfg)
        first, first_audit = selected_attention(query, cache, manifest, step_index=0, layer_index=0)
        self.assertEqual(first_audit["query_position"], 0)
        torch.testing.assert_close(first[0], cache.values[0, 0, 0])
        cache.write(0, 1, *row_pair(cfg, 0, 1))
        second, audit = selected_attention(query, cache, manifest, step_index=1, layer_index=0)
        self.assertEqual((audit["cache_capacity"], audit["cache_valid_length"], audit["query_position"]), (2, 2, 1))
        torch.testing.assert_close(second, dense_selected_reference(query, cache, manifest, 1, 0))


if __name__ == "__main__":
    unittest.main()
