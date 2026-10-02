"""CPU-only owned lifecycle/timing contracts. No GPU worker target is invoked."""
import ast
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from artifacts.janus_artifact import owned_gpu_timing as timing
from artifacts.janus_artifact.schema import ValidationError

UUID = "12345678-1234-5678-9abc-123456789abc"


def idle(**changes):
    snapshot = {"index": 0, "_uuid": UUID, "utilization_gpu_percent": 0,
                "memory_used_mib": 0, "compute_mode": "Default", "mig_mode_current": "Disabled"}
    snapshot.update(changes)
    return snapshot


def row(base=1000000, elapsed=0.001):
    return {"host_before_start_record_ns": base, "host_after_end_record_ns": base + 10000,
            "host_after_synchronize_ns": base + 20000, "event_elapsed_ms": elapsed,
            "host_completion_bracket_ns": 20000, "host_minus_event_slack_ns": 20000 - elapsed * 1e6}


def final(index, config):
    rows = []
    for step in range(config.steps):
        entry = row(1000000 + step * 100000)
        entry.update(sample_id=f"owned-worker-{index}-sample-{step}",
                     step_id=f"owned-worker-{index}-step-{step}", step_index=step,
                     exact_full_vector_equal=True,
                     nonfinite_values="impossible_for_int64_and_exact_bounded_oracle")
        rows.append(entry)
    pairs = []
    for pair in range(3):
        entry = row(100000 + pair * 100000, 0.0)
        entry["host_start_spacing_ns"] = None if pair == 0 else 100000
        pairs.append(entry)
    return {"kind": "final", "worker_id": f"owned-worker-{index}", "status": "ok",
            "tensor_dtype": "int64", "elements": config.elements,
            "ground_truth_all_passed": True, "steps": rows, "empty_event_pairs": pairs,
            "peak_allocated_bytes": 16 * config.elements,
            "peak_reserved_bytes": 16 * config.elements,
            "selected_operation_event_elapsed_ms_sum": sum(value["event_elapsed_ms"] for value in rows),
            "cleanup": {"completed": True, "allocated_bytes_after": 0, "reserved_bytes_after": 0}}


class FakeProcess:
    def __init__(self, index=0, code=0, terminate_works=True, kill_works=True):
        self.pid = 100 + index
        self.returncode = code
        self.stdin, self.stdout = io.BytesIO(), io.BytesIO()
        self.terminate_works, self.kill_works = terminate_works, kill_works
        self.actions = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.actions.append("terminate")
        if self.terminate_works:
            self.returncode = -15

    def kill(self):
        self.actions.append("kill")
        if self.kill_works:
            self.returncode = -9


class OwnedGPUTimingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.destination = Path(self.directory.name) / "fresh"
        self.config = timing.OwnedGPUTimingConfig(steps=2, elements=128, timeout_seconds=10)

    def run_fake(self, *, snapshots=None, processes=None, receive=None, spawn_error=None,
                 cleanup=None, wait=True):
        processes = processes or [FakeProcess(0), FakeProcess(1)]
        ready = [{"kind": "ready", "worker_id": f"owned-worker-{index}", "status": "ok",
                  "ordinal_scope": "single_device_unset_visibility"} for index in range(2)]
        snapshots = snapshots if snapshots is not None else [idle()] * 4
        receive = receive if receive is not None else [ready, [final(index, self.config) for index in range(2)]]
        with patch.dict(os.environ, {}, clear=True), patch.object(timing, "_query_gpu", side_effect=snapshots) as query, \
                patch.object(timing, "_spawn_worker", side_effect=spawn_error or processes) as spawn, \
                patch.object(timing, "_send_json") as send, \
                patch.object(timing, "_receive_stage", side_effect=receive) as receiver, \
                patch.object(timing, "_wait_exits", return_value=wait):
            if cleanup is None:
                report = timing.run_owned_gpu_timing(self.destination, self.config)
            else:
                with patch.object(timing, "_cleanup_workers", return_value=cleanup):
                    report = timing.run_owned_gpu_timing(self.destination, self.config)
        return report, query, spawn, send, receiver

    def test_config_default_budget_and_strict_limits(self):
        config = timing.OwnedGPUTimingConfig()
        config.validate()
        self.assertEqual(config.elements * timing.TENSOR_BYTES_PER_ELEMENT, 5 * 1024 ** 2)
        timing.OwnedGPUTimingConfig(steps=8, elements=1048576, timeout_seconds=24).validate()
        for changes in ({"steps": True}, {"steps": 9}, {"steps": 0}, {"elements": 1.0},
                        {"elements": 1048577}, {"elements": False}, {"timeout_seconds": True},
                        {"timeout_seconds": float("nan")}, {"timeout_seconds": float("inf")},
                        {"timeout_seconds": 25}, {"timeout_seconds": 0}, {"timeout_seconds": 10 ** 400}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                timing.OwnedGPUTimingConfig(**changes).validate()

    def test_invalid_config_cannot_query_or_spawn_or_create_output(self):
        with patch.object(timing, "_query_gpu") as query, patch.object(timing, "_spawn_worker") as spawn:
            with self.assertRaises(ValidationError):
                timing.run_owned_gpu_timing(self.destination, timing.OwnedGPUTimingConfig(elements=True))
        query.assert_not_called()
        spawn.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_existing_directory_is_preserved(self):
        self.destination.mkdir()
        sentinel = self.destination / "keep"
        sentinel.write_text("existing")
        with patch.object(timing, "_query_gpu") as query, self.assertRaises(ValidationError):
            timing.run_owned_gpu_timing(self.destination, self.config)
        query.assert_not_called()
        self.assertEqual(sentinel.read_text(), "existing")

    def test_no_parent_torch_import_or_cuda_initialization(self):
        tree = ast.parse(Path(timing.__file__).read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in names))
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_owned_worker")
        self.assertTrue(any(isinstance(node, ast.Import) and node.names[0].name == "torch" for node in ast.walk(worker)))

    def test_aggregate_parser_requires_single_unambiguous_known_device(self):
        valid = f"0, GPU-{UUID}, 0, 0, Default, Disabled\n"
        self.assertEqual(timing._parse_gpu_query(valid), idle())
        for bad in ("", valid + valid, valid.replace("0, GPU-", "1, GPU-", 1),
                    valid.replace(f"GPU-{UUID}", "N/A"), valid.replace(f"GPU-{UUID}", "Not Supported"),
                    valid.replace(", 0, 0,", ", N/A, 0,"), "0, 2, 0, 0, Default"):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                timing._parse_gpu_query(bad)

    def test_pytorch_uuid_object_is_stringified_without_weakening_smi_contract(self):
        class FakeCUuuid:
            def __str__(self):
                return UUID
        properties = type("Properties", (), {"uuid": FakeCUuuid()})()
        self.assertTrue(timing._verify_worker_uuid(properties, UUID))
        self.assertFalse(timing._verify_worker_uuid(object(), UUID))
        with self.assertRaises(ValidationError):
            timing._verify_worker_uuid(properties, "ffffffff-ffff-ffff-ffff-ffffffffffff")
        with self.assertRaises(ValidationError):
            timing._normalize_uuid(FakeCUuuid())

    def test_every_preflight_snapshot_is_idle_before_any_worker(self):
        for position in range(3):
            for change in ({"utilization_gpu_percent": 1}, {"memory_used_mib": 1},
                           {"compute_mode": "Exclusive_Process"}, {"mig_mode_current": "Enabled"},
                           {"mig_mode_current": "N/A"}):
                with self.subTest(position=position, change=change):
                    self.destination = Path(self.directory.name) / f"fresh-{position}-{len(list(Path(self.directory.name).iterdir()))}"
                    snapshots = [idle()] * position + [idle(**change)]
                    report, query, spawn, *_ = self.run_fake(snapshots=snapshots)
                    self.assertEqual(report["status"], "blocked")
                    self.assertEqual(query.call_count, position + 1)
                    spawn.assert_not_called()
                    self.assertEqual(report["cleanup"]["state"], "not_started")

    def test_visibility_mapping_set_even_empty_prevents_query_and_worker(self):
        for value in ("0", "", "GPU-other"):
            with self.subTest(value=value):
                self.destination = Path(self.directory.name) / f"visibility-{len(value)}"
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}), \
                        patch.object(timing, "_query_gpu") as query, patch.object(timing, "_spawn_worker") as spawn:
                    report = timing.run_owned_gpu_timing(self.destination, self.config)
                self.assertEqual(report["status"], "blocked")
                query.assert_not_called()
                spawn.assert_not_called()

    def test_device_identity_change_blocks_before_spawn(self):
        report, _, spawn, *_ = self.run_fake(snapshots=[idle(), idle(_uuid="other")])
        self.assertEqual(report["status"], "blocked")
        spawn.assert_not_called()

    def test_unknown_query_failure_has_private_failed_diagnostics_without_workers(self):
        report, _, spawn, *_ = self.run_fake(snapshots=[ValidationError("unknown aggregate")])
        self.assertEqual(report["status"], "blocked")
        spawn.assert_not_called()
        self.assertEqual(json.loads((self.destination / "report.json").read_text())["status"], "blocked")
        self.assertEqual(self.destination.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.destination / "report.json").stat().st_mode & 0o777, 0o600)

    def test_success_requires_two_full_reports_exit_and_post_idle(self):
        report, query, spawn, send, receiver = self.run_fake()
        self.assertEqual((report["status"], report["cleanup"]["state"]), ("ok", "verified"))
        self.assertEqual((query.call_count, spawn.call_count, send.call_count, receiver.call_count), (4, 2, 4, 2))
        self.assertEqual(report["resource_budget"]["tensor_bytes"], 80 * self.config.elements)
        self.assertEqual(report["resource_budget"]["cpu_buffers_plus_owned_cuda_peak_reserved_bytes"], 80 * self.config.elements)
        self.assertEqual(len(report["workers"]), 2)
        self.assertEqual(report["actual_cross_context_gpu_overlap"], "unknown")
        self.assertIsNone(report["cpu_gpu_absolute_offset"])
        self.assertFalse(report["non_root_verified"])
        self.assertFalse(report["non_root_deployment_verified"])
        self.assertTrue(report["host_launch_overlap"]["observed"])
        self.assertNotIn(UUID, json.dumps(report))
        self.assertTrue(report["wall_deadline_met"])

    def test_second_spawn_failure_reaps_first_owned_handle(self):
        first = FakeProcess(code=None)
        report, _, spawn, *_ = self.run_fake(spawn_error=[first, OSError("second spawn failed")])
        self.assertEqual(spawn.call_count, 2)
        self.assertEqual(first.actions, ["terminate"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cleanup"]["state"], "unknown")
        self.assertEqual(len(report["cleanup"]["owned_processes"]), 1)
        self.assertTrue(report["cleanup"]["owned_processes"][0]["confirmed_dead"])
        self.assertTrue(first.stdin.closed and first.stdout.closed)

    def test_report_receipt_without_process_exit_cannot_succeed(self):
        processes = [FakeProcess(0, code=None), FakeProcess(1, code=None)]
        report, *_ = self.run_fake(processes=processes, wait=False)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(all(process.actions == ["terminate"] for process in processes))
        self.assertTrue(all(state["confirmed_dead"] for state in report["cleanup"]["owned_processes"]))

    def test_nonzero_exit_despite_ok_report_fails(self):
        report, *_ = self.run_fake(processes=[FakeProcess(0, code=1), FakeProcess(1)])
        self.assertEqual(report["status"], "failed")

    def test_incomplete_ground_truth_or_bad_report_fails(self):
        ready = [{"kind": "ready", "worker_id": f"owned-worker-{i}", "status": "ok",
                  "ordinal_scope": "single_device_unset_visibility"} for i in range(2)]
        invalid = final(0, self.config)
        invalid["steps"].pop()
        report, *_ = self.run_fake(receive=[ready, [invalid, final(1, self.config)]])
        self.assertEqual(report["status"], "failed")
        self.assertIn("step count", report["diagnostics"][0]["error"])

    def test_post_memory_or_unknown_state_is_cleanup_unknown(self):
        for post in (idle(memory_used_mib=1), ValidationError("unreadable aggregate"), idle(_uuid="changed")):
            with self.subTest(post=post):
                self.destination = Path(self.directory.name) / f"post-{len(list(Path(self.directory.name).iterdir()))}"
                report, *_ = self.run_fake(snapshots=[idle()] * 3 + [post])
                self.assertEqual((report["status"], report["cleanup"]["state"]), ("failed", "unknown"))

    def test_unkillable_owned_process_confirmation_is_unknown(self):
        states = [{"worker_id": f"owned-worker-{index}", "pid": index + 100,
                   "confirmed_dead": index == 0, "exitcode": 0 if index == 0 else None} for index in range(2)]
        report, *_ = self.run_fake(cleanup=states)
        self.assertEqual((report["status"], report["cleanup"]["state"]), ("failed", "unknown"))

    def test_full_exact_groundtruth_timing_and_peak_schema_rejects_aliases(self):
        timing._validate_worker_report(final(0, self.config), 0, self.config)
        for field, value in (("ground_truth_all_passed", 1), ("elements", True),
                             ("peak_allocated_bytes", True), ("peak_reserved_bytes", -1),
                             ("selected_operation_event_elapsed_ms_sum", float("nan"))):
            malformed = final(0, self.config)
            malformed[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                timing._validate_worker_report(malformed, 0, self.config)
        for field, value in (("exact_full_vector_equal", 1), ("step_id", "wrong"),
                             ("step_index", False), ("event_elapsed_ms", float("inf")),
                             ("host_before_start_record_ns", 1.0), ("host_after_synchronize_ns", 0)):
            malformed = final(0, self.config)
            malformed["steps"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                timing._validate_worker_report(malformed, 0, self.config)

    def test_combined_cpu_and_allocator_peak_budget_is_guarded(self):
        ready = [{"kind": "ready", "worker_id": f"owned-worker-{i}", "status": "ok",
                  "ordinal_scope": "single_device_unset_visibility"} for i in range(2)]
        reports = [final(index, self.config) for index in range(2)]
        for report in reports:
            report["peak_reserved_bytes"] = timing.MAX_TENSOR_BYTES // 2
        report, *_ = self.run_fake(receive=[ready, reports])
        self.assertEqual(report["status"], "failed")
        self.assertIn("reservation", report["diagnostics"][0]["error"])

    def test_fragmented_frames_are_bounded_and_nonfinite_json_is_rejected(self):
        buffer = bytearray()
        self.assertEqual(timing._decode_frames(buffer, b'{"kind":"rea'), [])
        self.assertEqual(timing._decode_frames(buffer, b'dy"}\n'), [{"kind": "ready"}])
        self.assertEqual(buffer, b"")
        for invalid in (b"x\n", b"[]\n", b'{"value":NaN}\n', b"x" * (timing.MAX_IPC_BYTES + 1)):
            with self.subTest(invalid=invalid[:20]), self.assertRaises(ValidationError):
                timing._decode_frames(bytearray(), invalid)

    def test_partial_frame_and_blocked_stdin_obey_deadline(self):
        process = MagicMock()
        process.stdin.fileno.return_value = 7
        with patch.object(timing.os, "set_blocking"), patch.object(timing.os, "write", side_effect=BlockingIOError), \
                patch.object(timing.select, "select", return_value=([], [], [])), \
                patch.object(timing.time, "monotonic", side_effect=[0.0, 0.01, 0.02, 0.04, 0.05]):
            with self.assertRaises(ValidationError):
                timing._send_json(process, {"kind": "go"}, 0.03)
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, "rb", buffering=0)
        self.addCleanup(stream.close)
        self.addCleanup(os.close, write_fd)
        os.write(write_fd, b'{"kind":"final"')
        process.stdout = stream
        process.poll.return_value = None
        begin = time.monotonic()
        with self.assertRaises(ValidationError):
            timing._receive_stage([process], [bytearray()], "final", begin + 0.03)
        self.assertLess(time.monotonic() - begin, 0.25)

    def test_cleanup_continues_after_signal_error_and_uses_kill(self):
        first, second = FakeProcess(0, code=None), FakeProcess(1, code=None, terminate_works=False)
        first.terminate = MagicMock(side_effect=PermissionError("owned signal failure"))
        with patch.object(timing.time, "sleep"), patch.object(timing.time, "monotonic", side_effect=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
            states = timing._cleanup_workers([first, second], 2)
        self.assertTrue(all(state["confirmed_dead"] for state in states))
        self.assertIn("terminate", states[0]["lifecycle_errors"])
        self.assertEqual(second.actions, ["terminate", "kill"])
        self.assertEqual(first.actions, ["kill"])

    def test_cleanup_poll_error_and_unkillable_state_are_not_claimed_dead(self):
        process = FakeProcess(code=None, terminate_works=False, kill_works=False)
        process.poll = MagicMock(side_effect=OSError("cannot verify owned handle"))
        with patch.object(timing.time, "sleep"), patch.object(timing.time, "monotonic", side_effect=[0, 1, 2, 3, 4]):
            states = timing._cleanup_workers([process], 1)
        self.assertFalse(states[0]["confirmed_dead"])
        self.assertIsNone(states[0]["exitcode"])
        self.assertEqual(process.actions, ["terminate", "kill"])

    def test_expired_deadline_does_not_add_fixed_wait(self):
        process = FakeProcess(code=None)
        with patch.object(timing.time, "monotonic", return_value=10), patch.object(timing.time, "sleep") as sleep:
            self.assertFalse(timing._wait_exits([process], 9))
        sleep.assert_not_called()
        with patch.object(timing.time, "monotonic", return_value=10):
            self.assertEqual(timing._remaining(9, 2), 0)

    def test_maximum_steps_report_uses_bounded_primitive_frames(self):
        config = timing.OwnedGPUTimingConfig(steps=8, elements=1048576)
        result = final(0, config)
        for entry in result["steps"] + result["empty_event_pairs"]:
            # Exercise realistic large monotonic integer timestamps without floats.
            for key in ("host_before_start_record_ns", "host_after_end_record_ns", "host_after_synchronize_ns"):
                entry[key] += 9223372030000000000
        output = io.StringIO()
        with patch.object(timing.sys, "stdout", output):
            timing._emit_worker(result)
        encoded = output.getvalue().encode()
        self.assertLessEqual(len(encoded), timing.MAX_IPC_BYTES)
        decoded = timing._decode_frames(bytearray(), encoded)
        self.assertEqual(len(decoded), 1)
        timing._validate_worker_report(decoded[0], 0, config)

    def test_query_eof_is_not_assumed_process_exit(self):
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, "rb", buffering=0)
        self.addCleanup(stream.close)
        os.write(write_fd, f"0, GPU-{UUID}, 0, 0, Default, Disabled\n".encode())
        os.close(write_fd)
        process = MagicMock()
        process.stdout = stream
        process.returncode = 0
        # stdout EOF is seen before the process's next poll confirms its exit.
        process.poll.side_effect = [None, 0, 0, 0]
        with patch.object(timing.subprocess, "Popen", return_value=process) as popen, \
                patch.object(timing.time, "sleep"):
            result = timing._query_gpu(time.monotonic() + 1)
        self.assertEqual(result, idle())
        process.kill.assert_not_called()
        command = popen.call_args.args[0]
        self.assertEqual(command, ["nvidia-smi", "--query-gpu=" + timing.SMI_FIELDS, "--format=csv,noheader,nounits"])
        self.assertNotIn("--query-compute-apps", " ".join(command))

    def test_exec_spawn_bootstrap_passes_only_worker_flag_to_module(self):
        fake = FakeProcess()
        with patch.object(timing.subprocess, "Popen", return_value=fake) as popen:
            self.assertIs(timing._spawn_worker(), fake)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[1], "-c")
        observed = []
        def capture(module, **kwargs):
            observed.append((module, list(sys.argv), kwargs))
        with patch.object(sys, "argv", ["-c"] + argv[3:]), patch.object(runpy, "run_module", side_effect=capture):
            exec(argv[2], {})
        self.assertEqual(observed[0][0], "artifacts.janus_artifact.owned_gpu_timing")
        self.assertEqual(observed[0][1], [observed[0][0], "--owned-worker"])
        self.assertEqual(observed[0][2], {"run_name": "__main__"})


if __name__ == "__main__":
    unittest.main()
