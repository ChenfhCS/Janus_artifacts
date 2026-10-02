"""CPU-only routing and failure reporting for the owned CUDA benchmark."""

import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

from artifacts.janus_artifact.cli import build_parser, main
from artifacts.janus_artifact.schema import ValidationError


class OwnedGPUTimingCLITests(unittest.TestCase):
    def invoke(self, arguments):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = main(arguments)
        return code, output.getvalue(), error.getvalue()

    def test_parser_has_small_defaults_without_resource_override_flags(self):
        parser = build_parser()
        args = parser.parse_args(["owned-gpu-timing-benchmark", "--work-dir", "new-output"])
        self.assertEqual((args.steps, args.elements, args.timeout_seconds), (3, 65536, 20.0))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["owned-gpu-timing-benchmark", "--work-dir", "new-output", "--force"])

    def test_cli_routes_explicit_config_and_preserves_unknown_gpu_overlap(self):
        report = {"status": "ok", "actual_cross_context_gpu_overlap": "unknown"}
        with patch("artifacts.janus_artifact.owned_gpu_timing.run_owned_gpu_timing", return_value=report) as call:
            code, output, error = self.invoke([
                "owned-gpu-timing-benchmark", "--work-dir", "new-output",
                "--steps", "2", "--elements", "128", "--timeout-seconds", "10",
            ])
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(json.loads(output), report)
        work_dir, config = call.call_args.args
        self.assertEqual(work_dir, "new-output")
        self.assertEqual((config.steps, config.elements, config.timeout_seconds), (2, 128, 10.0))

    def test_busy_or_cleanup_failure_is_nonzero_with_machine_readable_diagnostics(self):
        for status in ("blocked", "failed"):
            with self.subTest(status=status), patch(
                "artifacts.janus_artifact.owned_gpu_timing.run_owned_gpu_timing",
                return_value={"status": status, "actual_cross_context_gpu_overlap": "unknown"},
            ):
                code, output, error = self.invoke(["owned-gpu-timing-benchmark", "--work-dir", "new-output"])
                self.assertEqual((code, error), (2, ""))
                self.assertEqual(json.loads(output)["status"], status)

    def test_invalid_config_surfaces_validation_error_before_hardware_work(self):
        with patch("artifacts.janus_artifact.owned_gpu_timing.run_owned_gpu_timing", side_effect=ValidationError("bounded config")):
            code, output, error = self.invoke(["owned-gpu-timing-benchmark", "--work-dir", "new-output"])
        self.assertEqual((code, output), (2, ""))
        self.assertIn("bounded config", error)


if __name__ == "__main__":
    unittest.main()
