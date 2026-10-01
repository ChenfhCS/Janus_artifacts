"""Executable upstream CLI checks using only controlled synthetic data."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path

from artifacts.janus_artifact.cli import build_parser, main
from artifacts.janus_artifact.workload import WorkloadConfig


class UpstreamCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workload_config = self.root / "workload.json"
        self.workload_config.write_text(json.dumps(asdict(WorkloadConfig(case_count=12))))
        fixture_root = Path(__file__).resolve().parents[1] / "fixtures"
        self.reconstruction_config = fixture_root / "probe_reconstruction.json"

    def invoke(self, args):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = main(args)
        return code, output.getvalue(), error.getvalue()

    def generate(self):
        target = self.root / "workload-bundle.json"
        code, output, error = self.invoke([
            "upstream-generate", "--config", str(self.workload_config),
            "--output", str(target),
        ])
        self.assertEqual(code, 0, error)
        self.assertFalse(json.loads(output)["scientific_result"])
        return target

    def test_generate_and_reconstruct_preserve_explicit_ids_and_origin(self):
        bundle_path = self.generate()
        destination = self.root / "profiles.json"
        code, output, error = self.invoke([
            "upstream-reconstruct", "--input", str(bundle_path),
            "--config", str(self.reconstruction_config), "--output", str(destination),
        ])
        self.assertEqual(code, 0, error)
        summary = json.loads(output)
        self.assertEqual(summary["runs"], 12)
        self.assertEqual(summary["decoding_steps"], 36)
        self.assertEqual(summary["phase_source_kind"], "oracle_aligned_phase")
        bundle = json.loads(bundle_path.read_text())
        profiles = json.loads(destination.read_text())
        self.assertFalse(profiles["scientific_result"])
        self.assertEqual(
            {r["response_id"] for r in bundle["runs"]},
            {r["response_id"] for r in profiles["profiles"]},
        )
        expected = {s["step_id"] for r in bundle["runs"] for s in r["gold"]["steps"]}
        observed = {s["step_id"] for r in profiles["profiles"] for s in r["decoding_steps"]}
        self.assertEqual(expected, observed)

    def test_missing_calibration_fails_without_output(self):
        bundle_path = self.generate()
        obj = json.loads(bundle_path.read_text())
        del obj["runs"][0]["calibration"]
        bundle_path.write_text(json.dumps(obj))
        destination = self.root / "must-not-exist.json"
        code, _, error = self.invoke([
            "upstream-reconstruct", "--input", str(bundle_path),
            "--config", str(self.reconstruction_config), "--output", str(destination),
        ])
        self.assertEqual(code, 2)
        self.assertIn("calibration", error)
        self.assertFalse(destination.exists())

    def test_configs_are_required_and_smoke_defaults_to_cpu(self):
        parser = build_parser()
        parsed = parser.parse_args(["upstream-synthetic-smoke", "--work-dir", "generated"])
        self.assertEqual(parsed.device, "cpu")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["upstream-reconstruct", "--input", "recorded", "--output", "derived"])


if __name__ == "__main__":
    unittest.main()
