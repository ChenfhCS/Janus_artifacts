"""ATR CLI routing and strict manifest regressions; no real attack collection."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from artifacts.janus_artifact.cli import build_parser, main, run
from artifacts.janus_artifact.schema import load_jsonl, write_jsonl
from artifacts.tests.atr_fixture_helpers import synthetic_response


ROOT = Path(__file__).resolve().parents[1]


class ATRCLITests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "test-output"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="atr-cli-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def manifest(self, records):
        path = self.directory / "manifest.jsonl"
        write_jsonl(path, records)
        return path

    def invoke(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main([str(value) for value in argv])
        return status, stdout.getvalue(), stderr.getvalue()

    def infer_arguments(self, manifest, *additional):
        return build_parser().parse_args([
            "atr-infer", "--manifest", str(manifest),
            "--checkpoint-dir", str(self.directory / "checkpoint"),
            "--output", str(self.directory / "predictions.jsonl"),
            "--metrics-output", str(self.directory / "metrics.json"),
            *additional,
        ])

    def test_parser_exposes_all_atr_commands_and_defaults(self):
        commands = {
            "atr-validate": ["--manifest", "manifest"],
            "atr-split": ["--manifest", "manifest", "--output", "out"],
            "atr-freeze-vocabulary": ["--manifest", "manifest", "--output", "out"],
            "atr-train": ["--manifest", "manifest", "--config", "config", "--checkpoint-dir", "checkpoint"],
            "atr-infer": ["--manifest", "manifest", "--checkpoint-dir", "checkpoint", "--output", "out", "--metrics-output", "metrics"],
            "atr-synthetic-smoke": ["--work-dir", "work"],
        }
        for command, arguments in commands.items():
            with self.subTest(command=command):
                self.assertEqual(build_parser().parse_args([command, *arguments]).command, command)
        split = build_parser().parse_args(["atr-split", *commands["atr-split"]])
        self.assertEqual((split.train_ratio, split.validation_ratio, split.test_ratio), (0.8, 0.1, 0.1))
        self.assertEqual(split.seed, "janus-atr-v1")
        inference = build_parser().parse_args(["atr-infer", *commands["atr-infer"]])
        self.assertEqual((inference.split, inference.device, inference.no_evaluate), ("test", "auto", False))

    def test_validate_reports_complete_response_and_step_counts(self):
        manifest = self.manifest([synthetic_response()])
        status, output, error = self.invoke(["atr-validate", "--manifest", manifest])
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(json.loads(output), {"status": "ok", "responses": 1, "steps": 3})

    def test_validate_rejects_missing_alignment_tokenizer_and_real_provenance(self):
        for field in ("response_id", "alignment", "tokenizer"):
            record = synthetic_response()
            del record[field]
            with self.subTest(field=field):
                status, _, error = self.invoke(["atr-validate", "--manifest", self.manifest([record])])
                self.assertEqual(status, 2)
                self.assertIn("ATR response fields", error)
        record = synthetic_response()
        del record["tokenizer"]["revision"]
        status, _, error = self.invoke(["atr-validate", "--manifest", self.manifest([record])])
        self.assertEqual(status, 2)
        self.assertIn("tokenizer fields", error)
        record = synthetic_response()
        record["source_kind"] = "reconstructed_recorded"
        record["provenance"] = {}
        status, _, error = self.invoke(["atr-validate", "--manifest", self.manifest([record])])
        self.assertEqual(status, 2)
        self.assertIn("collection_run_id", error)

    def test_split_preserves_every_step_and_case_group(self):
        records = [synthetic_response(f"response-{index}", count=2, offset=index) for index in range(10)]
        records[1]["case_id"] = records[0]["case_id"]
        output = self.directory / "split.jsonl"
        status, _, error = self.invoke([
            "atr-split", "--manifest", self.manifest(records), "--output", output,
            "--train-ratio", "0.5", "--validation-ratio", "0.25", "--test-ratio", "0.25", "--seed", "cli-regression",
        ])
        self.assertEqual((status, error), (0, ""))
        assigned = load_jsonl(output)
        self.assertEqual(len(assigned), 10)
        self.assertEqual(sum(len(record["steps"]) for record in assigned), 20)
        by_response = {record["response_id"]: record for record in assigned}
        self.assertEqual(by_response["response-0"]["split"], by_response["response-1"]["split"])

    def test_freeze_vocabulary_excludes_test_tokens(self):
        training = synthetic_response("training", "train")
        testing = synthetic_response("testing", "test", offset=9)
        testing["steps"][0]["gold_token_id"] = 23
        output = self.directory / "vocabulary.json"
        status, result, error = self.invoke([
            "atr-freeze-vocabulary", "--manifest", self.manifest([training, testing]), "--output", output,
        ])
        self.assertEqual((status, error), (0, ""))
        vocabulary = json.loads(output.read_text())
        self.assertEqual(vocabulary["token_ids"], [11, 17])
        self.assertEqual(json.loads(result)["source_split"], "train")

    def test_train_routes_explicit_config_and_unvalidated_run_kind(self):
        from artifacts.janus_artifact.atr_features import ATRConfig

        manifest = self.manifest([synthetic_response()])
        checkpoint = self.directory / "checkpoint"
        config_path = ROOT / "fixtures/atr_resnet18_reconstruction.json"
        expected = json.loads(config_path.read_text())
        self.assertEqual(set(expected), set(ATRConfig.__dataclass_fields__))
        config = ATRConfig.from_json(config_path)
        self.assertEqual((config.architecture, config.base_channels, config.epochs), ("resnet18", 64, 20))
        self.assertEqual((config.sequential_augmentation, config.augmentation_strength), ("causal_running_mean", 0.5))
        with patch("artifacts.janus_artifact.atr.train_atr", return_value={"status": "ok"}) as train:
            status, _, error = self.invoke([
                "atr-train", "--manifest", manifest, "--config", config_path,
                "--checkpoint-dir", checkpoint, "--device", "cpu",
            ])
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(train.call_args.args[1], config)
        self.assertEqual(train.call_args.args[2], str(checkpoint))
        self.assertEqual(train.call_args.kwargs, {"device": "cpu", "run_kind": "unvalidated_research_run"})

    def test_infer_writes_every_step_including_missing_prediction_and_metrics(self):
        record = synthetic_response("testing", "test")
        predictions = [{"step_id": step["step_id"], "predicted_token_id": 11 if index < 2 else None}
                       for index, step in enumerate(record["steps"])]
        metrics = {"dasr": 1 / 3, "micro_denominator_all_gold_tokens": 3, "missing_predictions": 1}
        arguments = self.infer_arguments(self.manifest([record]), "--device", "cpu", "--split", "test")
        with patch("artifacts.janus_artifact.atr.predict_atr", return_value=(predictions, metrics)) as predict:
            result = run(arguments)
        self.assertEqual(load_jsonl(arguments.output), predictions)
        self.assertEqual(json.loads(Path(arguments.metrics_output).read_text()), metrics)
        self.assertEqual(result["predictions"], 3)
        self.assertEqual(result["evaluation_status"], "evaluated")
        self.assertEqual(predict.call_args.kwargs, {"split": "test", "device": "cpu", "evaluate": True})

    def test_no_evaluate_accepts_missing_gold_and_emits_explicit_marker(self):
        record = synthetic_response("testing", "test")
        for step in record["steps"]:
            step["gold_token_id"] = None
        record["steps"][2]["profile"] = None
        predictions = [{"step_id": step["step_id"], "predicted_token_id": None} for step in record["steps"]]
        arguments = self.infer_arguments(self.manifest([record]), "--no-evaluate")
        with patch("artifacts.janus_artifact.atr.predict_atr", return_value=(predictions, None)) as predict:
            result = run(arguments)
        self.assertFalse(predict.call_args.kwargs["evaluate"])
        self.assertEqual(len(load_jsonl(arguments.output)), 3)
        self.assertEqual(json.loads(Path(arguments.metrics_output).read_text()), {
            "status": "not_evaluated", "evaluation_enabled": False,
        })
        self.assertEqual(result["evaluation_status"], "not_evaluated")

    def test_evaluation_rejects_missing_gold_before_checkpoint_load(self):
        record = synthetic_response("testing", "test")
        record["steps"][0]["gold_token_id"] = None
        arguments = self.infer_arguments(self.manifest([record]))
        status, _, error = self.invoke([
            "atr-infer", "--manifest", arguments.manifest, "--checkpoint-dir", arguments.checkpoint_dir,
            "--output", arguments.output, "--metrics-output", arguments.metrics_output,
        ])
        self.assertEqual(status, 2)
        self.assertIn("requires gold token IDs", error)
        self.assertFalse(Path(arguments.output).exists())
        self.assertFalse(Path(arguments.metrics_output).exists())

    def test_smoke_routes_work_directory_and_device_without_claiming_science(self):
        report = {"status": "ok", "synthetic_only": True, "scientific_result": False}
        with patch("artifacts.janus_artifact.atr.run_synthetic_atr_smoke", return_value=report) as smoke:
            status, output, error = self.invoke([
                "atr-synthetic-smoke", "--work-dir", self.directory, "--device", "cpu",
            ])
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(json.loads(output), report)
        smoke.assert_called_once_with(str(self.directory), device="cpu")

    def test_existing_cli_does_not_require_atr_imports(self):
        blocked = {f"artifacts.janus_artifact.{module}": None for module in ("atr", "atr_data", "atr_features")}
        with patch.dict(sys.modules, blocked):
            status, _, error = self.invoke([
                "validate-traces", "--input", ROOT / "fixtures/synthetic_traces.jsonl",
            ])
        self.assertEqual((status, error), (0, ""))


if __name__ == "__main__":
    unittest.main()
