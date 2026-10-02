"""CPU-only metadata tests; no buffer allocation or hardware probe."""

import json
import unittest
from copy import deepcopy
from unittest import mock

from artifacts.janus_artifact import probe_boundary as boundary
from artifacts.janus_artifact.schema import ValidationError


class TensorLike:
    def __deepcopy__(self, memo):
        raise AssertionError("nonprimitive input must not be copied")

    def __int__(self):
        raise AssertionError("nonprimitive input must not be coerced")

    def __iter__(self):
        raise AssertionError("nonprimitive input must not be traversed")

    def __repr__(self):
        raise AssertionError("nonprimitive input must not be rendered")


class IntSubclass(int):
    pass


class StrSubclass(str):
    pass


class DictSubclass(dict):
    pass


class ListSubclass(list):
    pass


def request():
    return {
        "schema_version": boundary.SCHEMA_VERSION,
        "run_id": "a" * 32,
        "step_id": "b" * 32,
        "timing": {"host_not_before_ns": 0, "host_deadline_ns": 100},
        "buffers": [{"buffer_id": "c" * 32, "elements": 8, "dtype": "int64"}],
    }


class ProbeBoundaryTests(unittest.TestCase):
    def assert_invalid(self, value, message=None):
        with self.assertRaisesRegex(ValidationError, message or "."):
            boundary.validate_probe_request(value)

    def test_valid_output_is_independent_json_primitives(self):
        original = request()
        checked = boundary.validate_probe_request(original)
        self.assertEqual(checked, original)
        self.assertEqual(json.loads(json.dumps(checked, allow_nan=False)), original)
        self.assertIsNot(checked, original)
        self.assertIsNot(checked["timing"], original["timing"])
        self.assertIsNot(checked["buffers"], original["buffers"])
        self.assertIsNot(checked["buffers"][0], original["buffers"][0])
        checked["buffers"][0]["elements"] = 9
        checked["timing"]["host_not_before_ns"] = 1
        self.assertEqual(original["buffers"][0]["elements"], 8)
        self.assertEqual(original["timing"]["host_not_before_ns"], 0)
        original["buffers"].append({"buffer_id": "d" * 32, "elements": 1, "dtype": "float32"})
        self.assertEqual(len(checked["buffers"]), 1)

    def test_output_field_order_is_canonical_without_reordering_buffers(self):
        original = request()
        original["buffers"].append({"dtype": "float32", "elements": 3, "buffer_id": "d" * 32})
        original = dict(reversed(list(original.items())))
        checked = boundary.validate_probe_request(original)
        self.assertEqual(list(checked), ["schema_version", "run_id", "step_id", "timing", "buffers"])
        self.assertEqual(list(checked["timing"]), ["host_not_before_ns", "host_deadline_ns"])
        self.assertEqual(list(checked["buffers"][1]), ["buffer_id", "elements", "dtype"])
        self.assertEqual([item["buffer_id"] for item in checked["buffers"]], ["c" * 32, "d" * 32])

    def test_signed_int64_host_time_endpoint_is_supported(self):
        original = request()
        original["timing"] = {
            "host_not_before_ns": boundary.MAX_HOST_TIME_NS - 1,
            "host_deadline_ns": boundary.MAX_HOST_TIME_NS,
        }
        self.assertEqual(boundary.validate_probe_request(original), original)

    def test_four_maximum_buffers_are_valid_metadata_only(self):
        original = request()
        original["buffers"] = [
            {"buffer_id": f"{index:032x}", "elements": boundary.MAX_BUFFER_ELEMENTS, "dtype": "int64"}
            for index in range(4)
        ]
        checked = boundary.validate_probe_request(original)
        self.assertEqual(checked, original)
        self.assertEqual(sum(item["elements"] * 8 for item in checked["buffers"]), 32 * 1024 * 1024)

    def test_float32_and_minimum_elements_are_valid(self):
        original = request()
        original["buffers"][0].update(elements=1, dtype="float32")
        self.assertEqual(boundary.validate_probe_request(original), original)

    def test_forbidden_metadata_injection_at_every_object_layer(self):
        fields = (
            "selector_positions", "mask", "gold", "indices", "model_config",
            "tensors", "paths", "kwargs", "user_labels", "oracle", "manifest",
            "reload_latency_ns", "gpu_global_clock_ns",
        )
        for layer in ("request", "timing", "buffer"):
            for field in fields:
                with self.subTest(layer=layer, field=field):
                    original = request()
                    target = original if layer == "request" else (
                        original["timing"] if layer == "timing" else original["buffers"][0]
                    )
                    target[field] = TensorLike()
                    self.assert_invalid(original, "fields must equal")

    def test_missing_fields_are_rejected_at_every_object_layer(self):
        for layer in ("request", "timing", "buffer"):
            original = request()
            target = original if layer == "request" else (
                original["timing"] if layer == "timing" else original["buffers"][0]
            )
            for field in tuple(target):
                with self.subTest(layer=layer, field=field):
                    mutated = deepcopy(original)
                    section = mutated if layer == "request" else (
                        mutated["timing"] if layer == "timing" else mutated["buffers"][0]
                    )
                    del section[field]
                    self.assert_invalid(mutated, "fields must equal")

    def test_nonstring_object_keys_are_rejected(self):
        for layer in ("request", "timing", "buffer"):
            with self.subTest(layer=layer):
                original = request()
                target = original if layer == "request" else (
                    original["timing"] if layer == "timing" else original["buffers"][0]
                )
                target[1] = "extra"
                self.assert_invalid(original, "fields must equal")
        original = request()
        value = original.pop("run_id")
        original[StrSubclass("run_id")] = value
        self.assert_invalid(original, "fields must equal")

    def test_schema_version_is_exact_builtin_string(self):
        for value in ("janus.probe.request.v2", "", None, True, TensorLike(), StrSubclass(boundary.SCHEMA_VERSION)):
            with self.subTest(value_type=type(value).__name__):
                original = request()
                original["schema_version"] = value
                self.assert_invalid(original, "schema_version")

    def test_opaque_id_syntax_rejects_paths_labels_and_nonprimitive_values(self):
        invalid = (
            "A" * 32, "c" * 31, "c" * 33, "g" * 32, "/tmp/synthetic-model-path",
            "healthcare-case-57", "α" * 32, "c" * 32 + "\n", b"c" * 32,
            True, 10, None, TensorLike(), StrSubclass("c" * 32),
        )
        for field in ("run_id", "step_id", "buffer_id"):
            for value in invalid:
                with self.subTest(field=field, value_type=type(value).__name__):
                    original = request()
                    target = original["buffers"][0] if field == "buffer_id" else original
                    target[field] = value
                    self.assert_invalid(original, "lowercase hexadecimal ID")

    def test_request_and_nested_objects_must_be_builtin_dicts(self):
        for value in (None, [], (), TensorLike(), DictSubclass(request())):
            with self.subTest(value_type=type(value).__name__):
                self.assert_invalid(value, "builtin object")
        for field in ("timing", "buffer"):
            for value in (None, [], (), TensorLike(), DictSubclass()):
                with self.subTest(field=field, value_type=type(value).__name__):
                    original = request()
                    if field == "timing":
                        original["timing"] = value
                    else:
                        original["buffers"][0] = value
                    self.assert_invalid(original, "builtin object")

    def test_buffers_must_be_builtin_list_with_one_to_four_entries(self):
        for value in (None, (), TensorLike(), [], ListSubclass(request()["buffers"]), request()["buffers"] * 5):
            with self.subTest(value_type=type(value).__name__):
                original = request()
                original["buffers"] = value
                self.assert_invalid(original, "builtin array")

    def test_host_times_reject_float_bool_negative_overflow_and_objects(self):
        invalid = (
            0.5, 0.0, True, False, -1, 2**63, 2**100, float("nan"),
            float("inf"), "1", None, TensorLike(), IntSubclass(1),
        )
        for field in ("host_not_before_ns", "host_deadline_ns"):
            for value in invalid:
                with self.subTest(field=field, value_type=type(value).__name__):
                    original = request()
                    original["timing"][field] = value
                    self.assert_invalid(original, "builtin integer")

    def test_host_deadline_must_strictly_follow_start(self):
        for start, deadline in ((0, 0), (100, 100), (100, 99), (boundary.MAX_HOST_TIME_NS, boundary.MAX_HOST_TIME_NS)):
            with self.subTest(start=start, deadline=deadline):
                original = request()
                original["timing"] = {"host_not_before_ns": start, "host_deadline_ns": deadline}
                self.assert_invalid(original, "greater than")

    def test_elements_reject_nonbuiltin_integers_and_outside_bounds(self):
        for value in (0, -1, boundary.MAX_BUFFER_ELEMENTS + 1, 2**100, 1.0, True, False, None, TensorLike(), IntSubclass(1)):
            with self.subTest(value_type=type(value).__name__):
                original = request()
                original["buffers"][0]["elements"] = value
                self.assert_invalid(original, "elements.*builtin integer")

    def test_dtype_is_exact_supported_builtin_string(self):
        for value in ("float64", "INT64", "float", "<i8", 8, None, TensorLike(), StrSubclass("int64")):
            with self.subTest(value_type=type(value).__name__):
                original = request()
                original["buffers"][0]["dtype"] = value
                self.assert_invalid(original, "dtype must equal")

    def test_duplicate_buffer_ids_are_rejected_even_with_different_dtype(self):
        original = request()
        original["buffers"].append({"buffer_id": "c" * 32, "elements": 1, "dtype": "float32"})
        self.assert_invalid(original, "duplicate buffer_id")

    def test_total_byte_budget_is_checked_without_allocating_any_buffer(self):
        original = request()
        original["buffers"][0]["elements"] = 2
        with mock.patch.object(boundary, "MAX_OWN_BUFFER_BYTES", 16):
            self.assertEqual(boundary.validate_probe_request(original), original)
            original["buffers"].append({"buffer_id": "d" * 32, "elements": 1, "dtype": "float32"})
            self.assert_invalid(original, "byte budget exceeded")

    def test_shape_and_stride_cannot_expand_the_descriptor_interface(self):
        for field in ("shape", "strides", "offset", "data", "device"):
            with self.subTest(field=field):
                original = request()
                original["buffers"][0][field] = [2, 4]
                self.assert_invalid(original, "fields must equal")

    def test_config_manifest_and_oracle_bundle_cannot_be_used_as_request(self):
        manifest = {
            "schema_version": "janus.selector.replay.v1",
            "config": {"model": "synthetic"},
            "oracle": {"selected_positions": [1, 3]},
            "probe_request": request(),
        }
        self.assert_invalid(manifest, "fields must equal")
        for field in ("config", "manifest", "oracle"):
            original = request()
            original[field] = manifest
            self.assert_invalid(original, "fields must equal")

    def test_cyclic_or_tensorlike_data_is_rejected_without_traversal_or_copy(self):
        original = request()
        cycle = []
        cycle.append(cycle)
        original["buffers"] = cycle
        self.assert_invalid(original, "builtin object")
        original = request()
        original["timing"] = original
        self.assert_invalid(original, "fields must equal")
        original = request()
        original["buffers"][0]["elements"] = TensorLike()
        self.assert_invalid(original, "builtin integer")


if __name__ == "__main__":
    unittest.main()
