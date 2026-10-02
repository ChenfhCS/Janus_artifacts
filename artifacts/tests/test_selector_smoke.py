"""Actual CPU selector/static-cache integration and oracle/request separation."""

import json
from pathlib import Path
import stat
import tempfile
import unittest

from artifacts.janus_artifact.schema import ValidationError
from artifacts.janus_artifact.selector_smoke import run_selector_replay_smoke
from artifacts.janus_artifact.probe_boundary import validate_probe_request


class SelectorSmokeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_actual_cpu_closed_loop_and_oracle_request_separation(self):
        destination = self.root / "fresh"
        report = run_selector_replay_smoke(destination)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["device"], "cpu")
        self.assertEqual((report["generation_steps"], report["layer_step_evaluations"], report["query_head_evaluations"]), (3, 6, 24))
        self.assertTrue(report["all_references_close"])
        self.assertTrue(report["all_gathers_strict_subset_of_valid_prefix"])
        self.assertLessEqual(report["max_absolute_error"], 1e-12)
        self.assertTrue(report["static_cache_storage_unchanged"])
        self.assertEqual(report["static_cache_tensor_bytes"], 1152)
        self.assertFalse(report["real_model_loaded"])
        self.assertFalse(report["probe_executed"])
        self.assertFalse(report["physical_cacheline_access_verified"])
        self.assertFalse(report["scientific_result"])
        requests = json.loads((destination / "probe-request-contracts.json").read_text())
        oracle = json.loads((destination / "oracle-sidecar.json").read_text())
        self.assertEqual(len(requests["requests"]), 3)
        self.assertTrue(requests["contract_checks_only"])
        for request in requests["requests"]:
            validate_probe_request(request)
            self.assertEqual(set(request), {"schema_version", "run_id", "step_id", "timing", "buffers"})
            for forbidden in ("manifest_sha256", "target_token_id", "absolute_kv_positions", "prompt_token_ids", "scores"):
                self.assertNotIn(forbidden, json.dumps(request))
        self.assertEqual(oracle["manifest_sha256"], report["manifest_sha256"])
        self.assertEqual({r["step_id"] for r in requests["requests"]}, {e["step_id"] for e in oracle["evaluations"]})
        self.assertEqual(json.loads((destination / "report.json").read_text()), report)

    def test_existing_output_directory_and_file_are_preserved(self):
        directory = self.root / "existing"
        directory.mkdir()
        sentinel = directory / "keep.txt"
        sentinel.write_text("user data")
        with self.assertRaises(ValidationError):
            run_selector_replay_smoke(directory)
        self.assertEqual(sentinel.read_text(), "user data")
        self.assertEqual(list(directory.iterdir()), [sentinel])

    def test_runtime_reports_are_created_with_private_modes(self):
        directory = self.root / "private"
        run_selector_replay_smoke(directory)
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        for path in directory.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, path.name)


if __name__ == "__main__":
    unittest.main()
