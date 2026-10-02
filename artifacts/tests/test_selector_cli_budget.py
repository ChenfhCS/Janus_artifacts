"""Selector-specific bounded-read regressions; tiny fixtures and sparse files."""

import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from unittest.mock import patch

from artifacts.janus_artifact import cli
from artifacts.janus_artifact.schema import ValidationError
from artifacts.janus_artifact.selector_replay import SelectorReplayConfig, build_selector_manifest


ROOT = Path(__file__).resolve().parents[1]


class ReadSpy:
    def __init__(self, handle, on_read=None):
        self.handle = handle
        self.calls = []
        self.total = 0
        self.on_read = on_read

    def _read(self, method, size):
        if size < 0:
            raise AssertionError("unbounded input read")
        self.calls.append((method, size))
        value = getattr(self.handle, method)(size)
        self.total += len(value)
        if self.on_read is not None:
            callback, self.on_read = self.on_read, None
            callback()
        return value

    def read(self, size=-1):
        return self._read("read", size)

    def readline(self, size=-1):
        return self._read("readline", size)

    def fileno(self):
        return self.handle.fileno()

    def __enter__(self):
        self.handle.__enter__()
        return self

    def __exit__(self, *args):
        return self.handle.__exit__(*args)


class SelectorCLIBudgetTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "test-output"
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="selector-budget-", dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = SelectorReplayConfig(
            run_id="1" * 32, model_id="synthetic", model_revision="v1",
            tokenizer_id="synthetic", tokenizer_revision="v1",
            model_config_sha256="2" * 64, weights_sha256=None,
            source_kind="synthetic_tensor_fixture", seed=19,
            prompt_token_ids=[1, 2], teacher_forced_token_ids=[3, 4],
            layers=1, query_heads=1, kv_heads=1, head_dim=2, top_k=1,
        )
        self.config_path = self.directory / "config.json"
        self.config_path.write_text(json.dumps(self.config.to_dict()))
        self.scores = self.directory / "scores.jsonl"
        self.rows = [
            {"step_index": 0, "layer_index": 0, "query_head_index": 0, "scores": [0.2, 0.8]},
            {"step_index": 1, "layer_index": 0, "query_head_index": 0, "scores": [0.2, 0.3, 0.5]},
        ]
        self.output = self.directory / "manifest.json"

    def invoke(self, command, path):
        if command == "selector-build":
            arguments = [command, "--config", self.config_path, "--scores", path, "--output", self.output]
        else:
            arguments = [command, "--expected-config", self.config_path, "--manifest", path]
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(SelectorReplayConfig, "from_json", return_value=self.config), redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main([str(value) for value in arguments])
        return status, stdout.getvalue(), stderr.getvalue()

    @contextmanager
    def spies(self, on_read=None):
        original_fdopen, original_loads = os.fdopen, json.loads
        streams = []

        def opened(*args, **kwargs):
            stream = ReadSpy(original_fdopen(*args, **kwargs), on_read)
            streams.append(stream)
            return stream

        with patch.object(cli.os, "fdopen", side_effect=opened), patch.object(cli.json, "loads", wraps=original_loads) as parse:
            yield streams, parse

    def understated(self, path, size=0):
        info = path.stat()
        return types.SimpleNamespace(**{key: size if key == "st_size" else getattr(info, key)
                                       for key in ("st_dev", "st_ino", "st_size", "st_mode", "st_mtime_ns", "st_ctime_ns")})

    def assert_refused(self, result, error):
        status, output, message = result
        self.assertEqual((status, output), (2, ""))
        self.assertIn(error, message)
        self.assertFalse(self.output.exists())

    def test_normal_build_and_validate_preserve_real_contract(self):
        self.scores.write_text("\n" + "\n".join(json.dumps(row) for row in self.rows) + "\n \n")
        result = self.invoke("selector-build", self.scores)
        self.assertEqual(result[0], 0, result[2])
        built = json.loads(self.output.read_text())
        self.assertEqual(len(built["steps"]), 2)
        result = self.invoke("selector-validate", self.output)
        self.assertEqual(result[0], 0, result[2])
        self.assertEqual(json.loads(result[1])["manifest_sha256"], built["manifest_sha256"])

    def test_default_sparse_scores_file_budget_refuses_before_open_or_parse(self):
        with self.scores.open("wb") as handle:
            handle.truncate(cli.MAX_SELECTOR_SCORES_FILE_BYTES + 1)
        with self.spies() as (streams, parse):
            result = self.invoke("selector-build", self.scores)
        self.assert_refused(result, "byte budget")
        self.assertEqual(streams, [])
        parse.assert_not_called()

    def test_default_sparse_manifest_budget_refuses_before_open_or_parse(self):
        path = self.directory / "large-manifest.json"
        with path.open("wb") as handle:
            handle.truncate(cli.MAX_SELECTOR_MANIFEST_FILE_BYTES + 1)
        with self.spies() as (streams, parse):
            result = self.invoke("selector-validate", path)
        self.assert_refused(result, "byte budget")
        self.assertEqual(streams, [])
        parse.assert_not_called()

    def test_default_physical_line_budget_does_not_read_all_or_parse(self):
        self.scores.write_bytes(b" " * (2 * cli.MAX_SELECTOR_SCORE_LINE_BYTES) + b"\n")
        with self.spies() as (streams, parse):
            result = self.invoke("selector-build", self.scores)
        self.assert_refused(result, "physical line byte budget")
        parse.assert_not_called()
        self.assertEqual(streams[0].total, cli.MAX_SELECTOR_SCORE_LINE_BYTES + 1)
        self.assertLess(streams[0].total, self.scores.stat().st_size)

    def test_line_budget_counts_newline_and_accepts_exact_cap_at_eof(self):
        with patch.object(cli, "MAX_SELECTOR_SCORE_LINE_BYTES", 8):
            self.scores.write_bytes(b"{}" + b" " * 6)
            self.assertEqual(cli._load_selector_score_rows(str(self.scores), 1), [{}])
            self.scores.write_bytes(b"{}" + b" " * 6 + b"\n")
            with self.spies() as (_, parse), self.assertRaisesRegex(ValidationError, "physical line"):
                cli._load_selector_score_rows(str(self.scores), 1)
            parse.assert_not_called()

    def test_excess_nonempty_row_refused_before_parsing_invalid_next_row(self):
        self.scores.write_text("\n".join(json.dumps(row) for row in self.rows) + "\n \nnot-json\n")
        with self.spies() as (_, parse):
            result = self.invoke("selector-build", self.scores)
        self.assert_refused(result, "row budget")
        self.assertEqual(parse.call_count, 2)

    def test_descriptor_budget_refuses_understated_path_stat_before_read(self):
        self.scores.write_bytes(b" " * 32)
        fake = self.understated(self.scores)
        with patch.object(cli, "MAX_SELECTOR_SCORES_FILE_BYTES", 16), patch.object(cli.Path, "stat", return_value=fake), self.spies() as (streams, parse):
            result = self.invoke("selector-build", self.scores)
        self.assert_refused(result, "byte budget")
        self.assertEqual(streams, [])
        parse.assert_not_called()

    def test_scores_cumulative_budget_defeats_false_stats_and_counts_blank_lines(self):
        self.scores.write_bytes(b"\n" * 64)
        fake = self.understated(self.scores)
        with patch.object(cli, "MAX_SELECTOR_SCORES_FILE_BYTES", 16), patch.object(cli.Path, "stat", return_value=fake), patch.object(cli.os, "fstat", return_value=fake), self.spies() as (streams, parse):
            result = self.invoke("selector-build", self.scores)
        self.assert_refused(result, "byte budget")
        parse.assert_not_called()
        self.assertEqual(streams[0].total, 17)
        self.assertTrue(all(0 < size <= 17 for _, size in streams[0].calls))

    def test_manifest_bounded_read_defeats_false_stats_before_parse(self):
        path = self.directory / "false-size.json"
        path.write_bytes(b" " * 64)
        fake = self.understated(path)
        with patch.object(cli, "MAX_SELECTOR_MANIFEST_FILE_BYTES", 16), patch.object(cli.Path, "stat", return_value=fake), patch.object(cli.os, "fstat", return_value=fake), self.spies() as (streams, parse):
            result = self.invoke("selector-validate", path)
        self.assert_refused(result, "byte budget")
        parse.assert_not_called()
        self.assertEqual(streams[0].calls, [("read", 17)])
        self.assertEqual(streams[0].total, 17)

    def test_manifest_growth_after_read_is_refused_before_parse(self):
        path = self.directory / "growing.json"
        path.write_bytes(b"{}")

        def grow():
            with path.open("ab") as handle:
                handle.write(b" ")

        with self.spies(on_read=grow) as (_, parse):
            result = self.invoke("selector-validate", path)
        self.assert_refused(result, "changed during bounded read")
        parse.assert_not_called()

    def test_underestimated_size_within_budget_is_still_refused(self):
        path = self.directory / "false-small.json"
        path.write_bytes(b"{}")
        fake = self.understated(path)
        with patch.object(cli.Path, "stat", return_value=fake), patch.object(cli.os, "fstat", return_value=fake), self.spies() as (_, parse):
            result = self.invoke("selector-validate", path)
        self.assert_refused(result, "changed during bounded read")
        parse.assert_not_called()

    def test_nonregular_scores_file_is_refused_without_opening_fifo(self):
        os.mkfifo(self.scores)
        with self.spies() as (streams, parse):
            result = self.invoke("selector-build", self.scores)
        self.assert_refused(result, "regular file")
        self.assertEqual(streams, [])
        parse.assert_not_called()

    def test_strict_scores_json_failures_return_exit_two_without_output(self):
        fixtures = [b'\xff\n', b'{"a":1,"a":2}\n', b'{"a":NaN}\n',
                    b'{"a":Infinity}\n', b'{"a":1e99999}\n',
                    b'{"a":' + b'[' * 2000 + b'0' + b']' * 2000 + b'}\n']
        for raw in fixtures:
            with self.subTest(raw=raw[:24]):
                self.scores.write_bytes(raw)
                self.assert_refused(self.invoke("selector-build", self.scores), "error:")

    def test_strict_manifest_json_failures_do_not_reach_validator(self):
        path = self.directory / "invalid.json"
        fixtures = [b'\xff', b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e99999}']
        for raw in fixtures:
            with self.subTest(raw=raw[:24]):
                path.write_bytes(raw)
                with patch("artifacts.janus_artifact.selector_replay.validate_selector_manifest") as validate:
                    self.assert_refused(self.invoke("selector-validate", path), "error:")
                validate.assert_not_called()

    def test_parser_errors_are_wrapped_as_validation_error(self):
        path = self.directory / "conversion.json"
        path.write_bytes(b"{}")
        for error in (ValueError("integer conversion limit"), RecursionError("JSON nesting limit")):
            with self.subTest(error=type(error).__name__):
                with patch.object(cli.json, "loads", side_effect=error):
                    self.assert_refused(self.invoke("selector-validate", path), "invalid selector manifest JSON")


    def compact_manifest_fixture(self):
        # A real valid contract with non-ASCII text makes byte/character counts differ.
        config = self.config.to_dict()
        config["model_id"] = "合成模型"
        self.config = SelectorReplayConfig.from_dict(config)
        self.scores.write_text("\n".join(json.dumps(row) for row in self.rows) + "\n")
        manifest = build_selector_manifest(self.config, self.rows)
        compact = (json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        return manifest, compact

    def test_compact_utf8_manifest_exact_budget_build_validate_roundtrip(self):
        manifest, compact = self.compact_manifest_fixture()
        pretty = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        self.assertGreater(len(pretty), len(compact))
        self.assertGreater(len(compact), len(compact.decode("utf-8")))
        self.assertTrue(compact.endswith(b"\n"))
        with patch.object(cli, "MAX_SELECTOR_MANIFEST_FILE_BYTES", len(compact)):
            result = self.invoke("selector-build", self.scores)
            self.assertEqual(result[0], 0, result[2])
            self.assertEqual(self.output.read_bytes(), compact)
            self.assertEqual(self.output.stat().st_size, len(compact))
            result = self.invoke("selector-validate", self.output)
            self.assertEqual(result[0], 0, result[2])
        self.assertEqual(json.loads(result[1])["manifest_sha256"], manifest["manifest_sha256"])

    def test_writer_overflow_one_byte_refused_before_creating_any_parent(self):
        _, compact = self.compact_manifest_fixture()
        self.output = self.directory / "new-parent" / "nested" / "manifest.json"
        with patch.object(cli, "MAX_SELECTOR_MANIFEST_FILE_BYTES", len(compact) - 1):
            result = self.invoke("selector-build", self.scores)
        self.assert_refused(result, "byte budget")
        self.assertFalse((self.directory / "new-parent").exists())

    def test_writer_streams_without_whole_manifest_dumps(self):
        manifest, compact = self.compact_manifest_fixture()
        with patch.object(cli.json, "dumps", side_effect=AssertionError("whole manifest dump")):
            cli._write_new_selector_json(str(self.output), manifest)
        self.assertEqual(self.output.read_bytes(), compact)

    def test_second_pass_budget_overflow_removes_only_owned_partial_output(self):
        self.compact_manifest_fixture()
        user = self.directory / "user-output.json"
        user.write_bytes(b"user data\n")
        with patch.object(cli, "MAX_SELECTOR_MANIFEST_FILE_BYTES", 3), patch.object(
            cli, "_selector_manifest_chunks", side_effect=[iter([b"{}\n"]), iter([b"{", b" " * 8, b"}\n"])],
        ):
            result = self.invoke("selector-build", self.scores)
        self.assert_refused(result, "byte budget")
        self.assertEqual(user.read_bytes(), b"user data\n")

    def test_second_pass_serialization_failure_removes_owned_partial_output(self):
        manifest, _ = self.compact_manifest_fixture()
        original = cli.json.JSONEncoder.iterencode
        calls = 0

        def encoding(encoder, value, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original(encoder, value, *args, **kwargs)

            def failing():
                yield "{"
                raise ValueError("second-pass serialization failed")
            return failing()

        with patch("artifacts.janus_artifact.selector_replay.build_selector_manifest", return_value=manifest), patch.object(cli.json.JSONEncoder, "iterencode", new=encoding):
            result = self.invoke("selector-build", self.scores)
        self.assert_refused(result, "finite JSON")

    def test_second_pass_write_failure_removes_owned_partial_output(self):
        self.compact_manifest_fixture()
        original = os.fdopen

        class FailingWrite:
            def __init__(self, handle):
                self.handle = handle
            def __enter__(self):
                self.handle.__enter__()
                return self
            def __exit__(self, *args):
                return self.handle.__exit__(*args)
            def write(self, raw):
                self.handle.write(raw[:1])
                raise OSError("synthetic write failure")

        def opened(descriptor, mode, **kwargs):
            handle = original(descriptor, mode, **kwargs)
            return FailingWrite(handle) if mode == "wb" else handle

        with patch.object(cli.os, "fdopen", side_effect=opened):
            result = self.invoke("selector-build", self.scores)
        self.assert_refused(result, "synthetic write failure")

    def test_writer_context_close_failure_removes_owned_partial_output(self):
        self.compact_manifest_fixture()
        original = os.fdopen

        class FailingClose:
            def __init__(self, handle):
                self.handle = handle
            def __enter__(self):
                self.handle.__enter__()
                return self
            def __exit__(self, *args):
                self.handle.__exit__(*args)
                raise OSError("synthetic close failure")
            def write(self, raw):
                return self.handle.write(raw)

        def opened(descriptor, mode, **kwargs):
            handle = original(descriptor, mode, **kwargs)
            return FailingClose(handle) if mode == "wb" else handle

        with patch.object(cli.os, "fdopen", side_effect=opened):
            result = self.invoke("selector-build", self.scores)
        self.assert_refused(result, "synthetic close failure")

    def test_writer_fdopen_failure_removes_owned_empty_output(self):
        self.compact_manifest_fixture()
        original = os.fdopen

        def opened(descriptor, mode, **kwargs):
            if mode == "wb":
                raise OSError("synthetic fdopen failure")
            return original(descriptor, mode, **kwargs)

        with patch.object(cli.os, "fdopen", side_effect=opened):
            result = self.invoke("selector-build", self.scores)
        self.assert_refused(result, "synthetic fdopen failure")

    def test_second_pass_failure_preserves_user_replacement_inode(self):
        manifest, _ = self.compact_manifest_fixture()
        original = cli._selector_manifest_chunks
        calls = 0

        def chunks(value):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original(value)

            def replaced():
                yield b"{"
                self.output.unlink()
                self.output.write_bytes(b"user replacement\n")
                raise ValidationError("synthetic serialization failure")
            return replaced()

        with patch.object(cli, "_selector_manifest_chunks", side_effect=chunks), self.assertRaisesRegex(ValidationError, "serialization failure"):
            cli._write_new_selector_json(str(self.output), manifest)
        self.assertEqual(self.output.read_bytes(), b"user replacement\n")


    def test_successful_second_pass_replacement_is_refused_and_preserved(self):
        _, compact = self.compact_manifest_fixture()
        replacement = b"u" * len(compact)  # Same size; inode identity must still fail.
        original = cli._selector_manifest_chunks
        calls = 0

        def chunks(value):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original(value)

            def replaced():
                iterator = original(value)
                yield next(iterator)
                self.output.unlink()
                self.output.write_bytes(replacement)
                yield from iterator
            return replaced()

        with patch.object(cli, "_selector_manifest_chunks", side_effect=chunks):
            status, output, error = self.invoke("selector-build", self.scores)
        self.assertEqual((status, output), (2, ""))
        self.assertIn("changed during bounded write", error)
        self.assertEqual(self.output.read_bytes(), replacement)


if __name__ == "__main__":
    unittest.main()
