"""CPU-only selector CLI routing and new-output preservation regressions."""

import io
import json
import stat
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from artifacts.janus_artifact.cli import build_parser, main
from artifacts.janus_artifact.schema import write_jsonl
from artifacts.janus_artifact.selector_replay import SelectorReplayConfig


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_MODULE = "artifacts.janus_artifact.selector_replay"
SMOKE_MODULE = "artifacts.janus_artifact.selector_smoke"


class SelectorCLITests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "test-output"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="selector-cli-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = SelectorReplayConfig(
            run_id="1" * 32, model_id="synthetic", model_revision="v1",
            tokenizer_id="synthetic", tokenizer_revision="v1",
            model_config_sha256="2" * 64, weights_sha256=None,
            source_kind="synthetic_tensor_fixture", seed=19,
            prompt_token_ids=[1, 2, 3], teacher_forced_token_ids=[4, 5],
            layers=1, query_heads=2, kv_heads=1, head_dim=2, top_k=1,
        )
        self.config_path = self.directory / "config.json"
        self.config_path.write_text(json.dumps(self.config.to_dict()))
        self.rows = [{"step_index": 0, "layer_index": 0, "query_head_index": 0, "scores": [0.2, 0.8]}]
        self.scores = self.directory / "scores.jsonl"
        write_jsonl(self.scores, self.rows)
        self.manifest = {
            "config": self.config.to_dict(), "steps": [{"step_index": 0}],
            "manifest_sha256": "3" * 64,
        }

    def invoke(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main([str(argument) for argument in arguments])
        return status, stdout.getvalue(), stderr.getvalue()

    def build_arguments(self, output):
        return ["selector-build", "--config", self.config_path, "--scores", self.scores, "--output", output]

    def smoke(self, report):
        module = types.ModuleType(SMOKE_MODULE)
        module.run_selector_replay_smoke = Mock(return_value=report)
        with patch.dict(sys.modules, {SMOKE_MODULE: module}):
            result = self.invoke(["selector-replay-smoke", "--work-dir", self.directory / "smoke"])
        return result, module.run_selector_replay_smoke

    def test_parser_requires_explicit_config_expected_config_and_paths(self):
        missing = [
            ["selector-build", "--scores", "scores", "--output", "out"],
            ["selector-build", "--config", "config", "--output", "out"],
            ["selector-build", "--config", "config", "--scores", "scores"],
            ["selector-validate", "--manifest", "manifest"],
            ["selector-validate", "--expected-config", "config"],
            ["selector-replay-smoke"],
        ]
        for arguments in missing:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                build_parser().parse_args(arguments)
            self.assertEqual(exc.exception.code, 2)

    def test_build_routes_explicit_config_and_jsonl_rows_to_new_manifest(self):
        output = self.directory / "manifest.json"
        with patch.object(SelectorReplayConfig, "from_json", return_value=self.config) as read_config, patch(
            f"{SELECTOR_MODULE}.build_selector_manifest", return_value=self.manifest
        ) as build:
            status, rendered, error = self.invoke(self.build_arguments(output))
        self.assertEqual((status, error), (0, ""))
        read_config.assert_called_once_with(str(self.config_path))
        build.assert_called_once_with(self.config, self.rows)
        self.assertEqual(json.loads(output.read_text()), self.manifest)
        result = json.loads(rendered)
        self.assertEqual((result["manifest_sha256"], result["steps"]), ("3" * 64, 1))
        self.assertEqual(result["source_kind"], "synthetic_tensor_fixture")
        self.assertFalse(result["scientific_result"])

    def test_build_creates_private_new_dirs_and_file_without_chmod_existing_parent(self):
        existing = self.directory / "existing"
        existing.mkdir(mode=0o750)
        original_mode = stat.S_IMODE(existing.stat().st_mode)
        output = existing / "new-a" / "new-b" / "manifest.json"
        with patch(f"{SELECTOR_MODULE}.build_selector_manifest", return_value=self.manifest):
            status, _, error = self.invoke(self.build_arguments(output))
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(stat.S_IMODE(existing.stat().st_mode), original_mode)
        self.assertEqual(stat.S_IMODE((existing / "new-a").stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(output.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_build_refuses_existing_user_file_and_preserves_bytes_permissions_and_mtime(self):
        output = self.directory / "user-manifest.json"
        contents = b"existing user output must survive\n"
        output.write_bytes(contents)
        output.chmod(0o640)
        original = output.stat()
        with patch(f"{SELECTOR_MODULE}.build_selector_manifest", return_value=self.manifest):
            status, rendered, error = self.invoke(self.build_arguments(output))
        self.assertEqual(status, 2)
        self.assertEqual(rendered, "")
        self.assertIn("File exists", error)
        self.assertEqual(output.read_bytes(), contents)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o640)
        self.assertEqual(output.stat().st_mtime_ns, original.st_mtime_ns)

    def test_build_refuses_existing_symlink_without_touching_its_target(self):
        target = self.directory / "user-target.json"
        target.write_bytes(b"user target\n")
        output = self.directory / "linked-output.json"
        output.symlink_to(target)
        with patch(f"{SELECTOR_MODULE}.build_selector_manifest", return_value=self.manifest):
            status, _, error = self.invoke(self.build_arguments(output))
        self.assertEqual(status, 2)
        self.assertIn("File exists", error)
        self.assertTrue(output.is_symlink())
        self.assertEqual(target.read_bytes(), b"user target\n")

    def test_invalid_config_is_rejected_before_build_and_output_creation(self):
        self.config_path.write_text("{}")
        output = self.directory / "must-not-create" / "manifest.json"
        with patch(f"{SELECTOR_MODULE}.build_selector_manifest") as build:
            status, rendered, error = self.invoke(self.build_arguments(output))
        self.assertEqual(status, 2)
        self.assertEqual(rendered, "")
        self.assertTrue(error.startswith("error:"))
        build.assert_not_called()
        self.assertFalse(output.parent.exists())

    def test_nonfinite_manifest_is_rejected_before_output_creation(self):
        manifest = {**self.manifest, "steps": [{"score": float("nan")} ]}
        output = self.directory / "must-not-create" / "manifest.json"
        with patch(f"{SELECTOR_MODULE}.build_selector_manifest", return_value=manifest):
            status, rendered, error = self.invoke(self.build_arguments(output))
        self.assertEqual((status, rendered), (2, ""))
        self.assertIn("finite JSON", error)
        self.assertFalse(output.parent.exists())

    def test_validate_routes_manifest_with_mandatory_expected_config(self):
        path = self.directory / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        with patch.object(SelectorReplayConfig, "from_json", return_value=self.config) as read_config, patch(
            f"{SELECTOR_MODULE}.validate_selector_manifest", return_value=self.manifest
        ) as validate:
            status, rendered, error = self.invoke([
                "selector-validate", "--manifest", path, "--expected-config", self.config_path,
            ])
        self.assertEqual((status, error), (0, ""))
        read_config.assert_called_once_with(str(self.config_path))
        validate.assert_called_once_with(self.manifest, expected_config=self.config)
        result = json.loads(rendered)
        self.assertEqual((result["status"], result["steps"]), ("ok", 1))
        self.assertFalse(result["scientific_result"])

    def test_smoke_is_cpu_only_route_without_device_or_gpu_arguments(self):
        parsed = build_parser().parse_args(["selector-replay-smoke", "--work-dir", "work"])
        self.assertFalse(hasattr(parsed, "device"))
        for flags in (["--device", "cpu"], ["--device", "cuda"], ["--gpu"], ["--auto"]):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                build_parser().parse_args(["selector-replay-smoke", "--work-dir", "work", *flags])
            self.assertEqual(exc.exception.code, 2)
        report = {"status": "ok", "device": "cpu", "scientific_result": False, "probe_executed": False}
        (status, rendered, error), smoke = self.smoke(report)
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(json.loads(rendered), report)
        smoke.assert_called_once_with(str(self.directory / "smoke"))

    def test_failed_smoke_returns_exit_two_with_finite_report(self):
        report = {"status": "failed", "device": "cpu", "scientific_result": False, "probe_executed": False}
        (status, rendered, error), _ = self.smoke(report)
        self.assertEqual((status, error), (2, ""))
        self.assertEqual(json.loads(rendered), report)

    def test_nonfinite_smoke_is_refused_without_printing_invalid_json(self):
        (status, rendered, error), _ = self.smoke({"status": "ok", "value": float("inf")})
        self.assertEqual((status, rendered), (2, ""))
        self.assertIn("finite JSON", error)

    def test_existing_cli_keeps_lazy_selector_imports(self):
        with patch.dict(sys.modules, {SELECTOR_MODULE: None, SMOKE_MODULE: None}):
            status, _, error = self.invoke([
                "validate-traces", "--input", ROOT / "fixtures/synthetic_traces.jsonl",
            ])
        self.assertEqual((status, error), (0, ""))


if __name__ == "__main__":
    unittest.main()
