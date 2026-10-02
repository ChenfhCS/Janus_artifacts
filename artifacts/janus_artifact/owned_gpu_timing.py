"""Bounded timing of two owned CUDA contexts; the parent imports only stdlib.

CUDA event times are context-local durations. They never establish a common
absolute GPU clock or actual cross-context overlap. No probe/model is run here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import select
import selectors
import subprocess
import sys
import time
import uuid
from typing import Any

from .schema import ValidationError

REPORT_VERSION = "janus.owned_gpu_timing.v1"
WALL_LIMIT_SECONDS = 30.0
WORKER_COUNT = 2
MAX_IPC_BYTES = 8192
MAX_TENSOR_BYTES = 1024 ** 3
# GPU input/output, CPU oracle input/output, CPU full result copy, per worker.
TENSOR_BYTES_PER_ELEMENT = 80
SMI_FIELDS = "index,uuid,utilization.gpu,memory.used,compute_mode,mig.mode.current"


@dataclass(frozen=True)
class OwnedGPUTimingConfig:
    steps: int = 3
    elements: int = 65536
    timeout_seconds: float = 20

    def validate(self) -> None:
        for name, maximum in (("steps", 8), ("elements", 1048576)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValidationError(f"{name} must be an integer in [1, {maximum}]")
        value = self.timeout_seconds
        if type(value) not in (int, float) or not 0 < value <= 24 or not math.isfinite(value):
            raise ValidationError("timeout_seconds must be finite and in (0, 24]")
        if self.elements * TENSOR_BYTES_PER_ELEMENT > MAX_TENSOR_BYTES:
            raise ValidationError("owned tensor budget exceeds 1 GiB")


def _remaining(deadline: float, maximum: float) -> float:
    return max(0.0, min(maximum, deadline - time.monotonic()))


def _finite_number(value: Any, name: str, *, nonnegative: bool = True) -> float:
    if type(value) not in (int, float):
        raise ValidationError(f"{name} must be a finite number")
    try:
        valid = math.isfinite(value) and (not nonnegative or value >= 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValidationError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return float(value)


def _host_clock() -> dict[str, Any]:
    info = time.get_clock_info("monotonic")
    previous = time.monotonic_ns()
    deltas = []
    for _ in range(256):
        current = time.monotonic_ns()
        if current > previous:
            deltas.append(current - previous)
        previous = current
    return {
        "name": "time.monotonic_ns", "implementation": info.implementation,
        "advertised_resolution_seconds": _finite_number(info.resolution, "host clock resolution"),
        "empirical_min_positive_delta_ns": min(deltas) if deltas else None,
        "empirical_sample_count": 256, "common_clock_scope": "same_host_only",
    }


def _normalize_uuid(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("GPU identity is unavailable")
    result = value.strip().lower()
    if result.startswith("gpu-"):
        result = result[4:]
    try:
        return str(uuid.UUID(result))
    except ValueError as exc:
        raise ValidationError("GPU identity format is unknown") from exc


def _verify_worker_uuid(properties: Any, expected_uuid: str) -> bool:
    observed_uuid = getattr(properties, "uuid", None)
    if observed_uuid is None:
        return False
    # PyTorch exposes _CUuuid, whose __str__ is canonical UUID text.
    if _normalize_uuid(str(observed_uuid)) != _normalize_uuid(expected_uuid):
        raise ValidationError("owned worker device identity differs from aggregate preflight")
    return True


def _parse_gpu_query(output: str) -> dict[str, Any]:
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValidationError("exactly one aggregate GPU row is required")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 6 or fields[0] != "0":
        raise ValidationError("GPU ordinal scope is ambiguous")
    uuid = _normalize_uuid(fields[1])
    for value in fields[2:4]:
        if not value.isascii() or not value.isdigit():
            raise ValidationError("GPU utilization/memory aggregate is unknown")
    return {
        "index": 0, "_uuid": uuid, "utilization_gpu_percent": int(fields[2]),
        "memory_used_mib": int(fields[3]), "compute_mode": fields[4],
        "mig_mode_current": fields[5],
    }


def _require_idle(snapshot: dict[str, Any]) -> None:
    if snapshot["compute_mode"] != "Default" or snapshot["mig_mode_current"] != "Disabled":
        raise ValidationError("Default compute mode and known disabled MIG are required")
    if snapshot["utilization_gpu_percent"] != 0 or snapshot["memory_used_mib"] != 0:
        raise ValidationError("GPU aggregate is busy or has allocated memory")


def _query_gpu(deadline: float) -> dict[str, Any]:
    """Aggregate only, with bounded reading, kill and exit confirmation."""
    duration = _remaining(deadline, 2.0)
    if duration <= 0.25:
        raise ValidationError("no time remains for bounded aggregate GPU query")
    query_deadline = min(deadline, time.monotonic() + duration)
    process = subprocess.Popen(
        ["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    content = bytearray()
    try:
        os.set_blocking(process.stdout.fileno(), False)
        # Reserve part of each <=2 s query budget for its own cleanup.
        read_deadline = query_deadline - min(0.2, duration / 4)
        while time.monotonic() < read_deadline:
            readable, _, _ = select.select([process.stdout], [], [], _remaining(read_deadline, 0.05))
            if readable:
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                content.extend(chunk)
                if len(content) > MAX_IPC_BYTES:
                    raise ValidationError("aggregate GPU query exceeded output budget")
            elif process.poll() is not None:
                break
        if not _wait_exits([process], read_deadline):
            raise ValidationError("aggregate GPU query timed out")
        if process.returncode != 0:
            raise ValidationError("aggregate GPU query failed")
        # Drain already available data after exit, without waiting for a frame.
        while True:
            try:
                chunk = os.read(process.stdout.fileno(), 4096)
            except BlockingIOError:
                break
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > MAX_IPC_BYTES:
                raise ValidationError("aggregate GPU query exceeded output budget")
        return _parse_gpu_query(content.decode("utf-8", errors="strict"))
    finally:
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=_remaining(query_deadline, 0.2))
            except subprocess.TimeoutExpired:
                pass
        dead = process.poll() is not None
        process.stdout.close()
        if not dead:
            raise ValidationError("aggregate GPU query process exit could not be confirmed")


def _public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if key != "_uuid"}


def _spawn_worker() -> subprocess.Popen:
    # An exec-spawn avoids shared-memory/resource-tracker helper processes.
    package = __package__
    import_root = str(Path(__file__).resolve().parents[len(package.split("."))])
    bootstrap = "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); module=sys.argv.pop(1); sys.argv[0]=module; runpy.run_module(module,run_name='__main__')"
    process = subprocess.Popen(
        [sys.executable, "-c", bootstrap, import_root, f"{package}.owned_gpu_timing", "--owned-worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return process


def _send_json(process: subprocess.Popen, value: dict[str, Any], deadline: float) -> None:
    frame = (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(frame) > MAX_IPC_BYTES:
        raise ValidationError("owned IPC send frame exceeds budget")
    fd = process.stdin.fileno()
    os.set_blocking(fd, False)
    offset = 0
    while offset < len(frame):
        if _remaining(deadline, 0.05) <= 0:
            raise ValidationError("owned IPC send timed out")
        try:
            offset += os.write(fd, frame[offset:])
        except BlockingIOError:
            select.select([], [fd], [], _remaining(deadline, 0.05))


def _decode_frames(buffer: bytearray, chunk: bytes) -> list[dict[str, Any]]:
    buffer.extend(chunk)
    if len(buffer) > MAX_IPC_BYTES:
        raise ValidationError("owned IPC frame exceeds budget")
    messages = []
    while b"\n" in buffer:
        frame, _, rest = buffer.partition(b"\n")
        buffer[:] = rest
        try:
            value = json.loads(frame, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        except (ValueError, UnicodeError) as exc:
            raise ValidationError("invalid owned IPC JSON") from exc
        if not isinstance(value, dict):
            raise ValidationError("owned IPC must contain primitive JSON objects")
        messages.append(value)
    return messages


def _receive_stage(processes: list[subprocess.Popen], buffers: list[bytearray],
                   kind: str, deadline: float) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(processes)
    with selectors.DefaultSelector() as selector:
        for index, process in enumerate(processes):
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ, index)
        while any(value is None for value in results):
            if _remaining(deadline, 0.05) <= 0:
                raise ValidationError(f"owned worker {kind} timed out")
            for key, _ in selector.select(_remaining(deadline, 0.05)):
                index = key.data
                chunk = os.read(key.fileobj.fileno(), 4096)
                if not chunk:
                    raise ValidationError(f"owned worker exited before complete {kind} report")
                for message in _decode_frames(buffers[index], chunk):
                    if results[index] is not None or message.get("kind") != kind:
                        raise ValidationError(f"unexpected owned worker message during {kind}")
                    results[index] = message
                    selector.unregister(key.fileobj)
            for index, process in enumerate(processes):
                if results[index] is None and process.poll() is not None:
                    # An exited process may still have its full final frame in the pipe;
                    # the selector gets one more nonblocking drain on the next pass.
                    readable, _, _ = select.select([process.stdout], [], [], 0)
                    if not readable:
                        raise ValidationError(f"owned worker exited without {kind} report")
    return results


def _wait_exits(processes: list[subprocess.Popen], deadline: float) -> bool:
    while any(process.poll() is None for process in processes):
        duration = _remaining(deadline, 0.02)
        if duration <= 0:
            return False
        time.sleep(duration)
    return True


def _cleanup_workers(processes: list[subprocess.Popen], deadline: float) -> list[dict[str, Any]]:
    errors: list[dict[str, str]] = [{} for _ in processes]

    def poll(index: int) -> tuple[bool, int | None]:
        try:
            code = processes[index].poll()
            return code is not None, code
        except OSError as exc:
            errors[index]["poll"] = f"{type(exc).__name__}: {str(exc)[:100]}"
            return False, None

    def wait_until(limit: float) -> None:
        while not all(poll(index)[0] for index in range(len(processes))):
            duration = _remaining(limit, 0.02)
            if duration <= 0:
                break
            time.sleep(duration)

    for index, process in enumerate(processes):
        if not poll(index)[0]:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors[index]["terminate"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    wait_until(min(deadline, time.monotonic() + 0.8))
    for index, process in enumerate(processes):
        if not poll(index)[0]:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors[index]["kill"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    wait_until(deadline)
    states = []
    for index, process in enumerate(processes):
        dead, code = poll(index)
        states.append({"worker_id": f"owned-worker-{index}", "pid": process.pid,
                       "confirmed_dead": dead, "exitcode": code,
                       "lifecycle_errors": errors[index]})
    return states


def _timing_row(row: Any, prefix: str) -> None:
    if not isinstance(row, dict):
        raise ValidationError(f"{prefix} must be an object")
    timestamps = [row.get(key) for key in (
        "host_before_start_record_ns", "host_after_end_record_ns", "host_after_synchronize_ns")]
    if any(type(value) is not int or value < 0 for value in timestamps) or timestamps != sorted(timestamps):
        raise ValidationError(f"{prefix} host timestamps must be ordered builtin integers")
    _finite_number(row.get("event_elapsed_ms"), f"{prefix} elapsed")
    bracket = timestamps[2] - timestamps[0]
    if row.get("host_completion_bracket_ns") != bracket or type(row.get("host_completion_bracket_ns")) is not int:
        raise ValidationError(f"{prefix} completion bracket mismatch")
    expected_slack = bracket - row["event_elapsed_ms"] * 1e6
    actual_slack = _finite_number(row.get("host_minus_event_slack_ns"), f"{prefix} slack", nonnegative=False)
    if not math.isclose(actual_slack, expected_slack, rel_tol=1e-12, abs_tol=1e-6):
        raise ValidationError(f"{prefix} host/event slack mismatch")


def _validate_worker_report(report: Any, index: int, config: OwnedGPUTimingConfig) -> None:
    if not isinstance(report, dict) or report.get("kind") != "final" or report.get("worker_id") != f"owned-worker-{index}":
        raise ValidationError("owned worker final identity mismatch")
    if report.get("status") != "ok" or report.get("ground_truth_all_passed") is not True:
        raise ValidationError("owned worker exact ground truth did not pass")
    if (report.get("tensor_dtype") != "int64" or type(report.get("elements")) is not int
            or report["elements"] != config.elements):
        raise ValidationError("owned worker tensor contract mismatch")
    rows = report.get("steps")
    if not isinstance(rows, list) or len(rows) != config.steps:
        raise ValidationError("owned worker step count mismatch")
    for step, row in enumerate(rows):
        _timing_row(row, f"worker {index} step {step}")
        if (row.get("sample_id") != f"owned-worker-{index}-sample-{step}"
                or row.get("step_id") != f"owned-worker-{index}-step-{step}"
                or type(row.get("step_index")) is not int or row["step_index"] != step
                or row.get("exact_full_vector_equal") is not True
                or row.get("nonfinite_values") != "impossible_for_int64_and_exact_bounded_oracle"):
            raise ValidationError("owned worker ground-truth/step identity contract mismatch")
    pairs = report.get("empty_event_pairs")
    if not isinstance(pairs, list) or len(pairs) != 3:
        raise ValidationError("owned worker empty-event measurement count mismatch")
    for pair, row in enumerate(pairs):
        _timing_row(row, f"worker {index} empty pair {pair}")
        spacing = row.get("host_start_spacing_ns")
        if spacing is not None and (type(spacing) is not int or spacing < 0):
            raise ValidationError("empty-event host spacing must be nonnegative integer or null")
    cleanup = report.get("cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("completed") is not True:
        raise ValidationError("owned worker allocator cleanup is unconfirmed")
    for key in ("allocated_bytes_after", "reserved_bytes_after"):
        if type(cleanup.get(key)) is not int or cleanup[key] != 0:
            raise ValidationError("owned worker retains allocated/reserved tensor memory")
    allocated, reserved = (report.get(key) for key in ("peak_allocated_bytes", "peak_reserved_bytes"))
    if (type(allocated) is not int or type(reserved) is not int
            or not config.elements * 16 <= allocated <= reserved <= MAX_TENSOR_BYTES):
        raise ValidationError("owned worker allocator peak budget is invalid")
    expected_sum = sum(row["event_elapsed_ms"] for row in rows)
    actual_sum = _finite_number(report.get("selected_operation_event_elapsed_ms_sum"), "selected operation sum")
    if not math.isclose(actual_sum, expected_sum, rel_tol=1e-12, abs_tol=1e-12):
        raise ValidationError("owned worker event sum mismatch")


def _host_overlap(reports: list[dict[str, Any]]) -> dict[str, Any]:
    intersections = []
    for first in reports[0]["steps"]:
        for second in reports[1]["steps"]:
            start = max(first["host_before_start_record_ns"], second["host_before_start_record_ns"])
            end = min(first["host_after_end_record_ns"], second["host_after_end_record_ns"])
            if end > start:
                intersections.append({"sample_ids": [first["sample_id"], second["sample_id"]],
                                      "intersection_ns": end - start})
    return {"observed": bool(intersections), "interval_intersections": intersections,
            "meaning": "host_submission_brackets_only; not GPU execution overlap"}


def run_owned_gpu_timing(work_dir: str | Path, config: OwnedGPUTimingConfig) -> dict[str, Any]:
    """Create a fresh diagnostic directory; never initialize CUDA in the parent."""
    entered = time.monotonic()
    hard_deadline = entered + WALL_LIMIT_SECONDS
    if not isinstance(config, OwnedGPUTimingConfig):
        raise ValidationError("config must be OwnedGPUTimingConfig")
    config.validate()
    destination = Path(work_dir)
    try:
        destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValidationError("work_dir must be new; existing files are preserved") from exc
    report: dict[str, Any] = {
        "schema_version": REPORT_VERSION, "status": "blocked", "config": asdict(config),
        "scientific_result_validated": False, "paper_reproduction": False,
        "actual_cross_context_gpu_overlap": "unknown", "cpu_gpu_absolute_offset": None,
        "non_root_verified": False, "non_root_deployment_verified": False,
        "execution_uid": os.getuid(), "execution_euid": os.geteuid(),
        "worker_count_requested": WORKER_COUNT, "workers": [], "preflight_snapshots": [],
        "resource_budget": {"tensor_bytes": config.elements * TENSOR_BYTES_PER_ELEMENT,
            "tensor_bytes_limit": MAX_TENSOR_BYTES, "vectors_per_worker": 5, "bytes_per_element": 8,
            "includes_cuda_context_or_allocator_overhead": False,
            "context_driver_overhead_device_bytes": "not_measured_as_separate_device_bytes",
            "cpu_buffer_bytes": config.elements * 48,
            "wall_limit_seconds": WALL_LIMIT_SECONDS, "report_write_margin_seconds": 0.5},
        "host_clock": _host_clock(), "event_clock": "relative CUDA-event elapsed_ms within each owned context",
        "empty_event_measurement_interpretation": "empirical intervals, not a known physical timer resolution",
        "cleanup": {"state": "not_started", "owned_processes": [], "post_gpu_aggregate": None},
        "host_launch_overlap": None, "diagnostics": [],
    }
    processes: list[subprocess.Popen] = []
    results = []
    expected_uuid = None
    preflight_passed = False
    compute_deadline = min(entered + config.timeout_seconds, hard_deadline - 6.5)
    try:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            raise ValidationError("CUDA_VISIBLE_DEVICES mapping must be unset for this single-device benchmark")
        for _ in range(3):
            snapshot = _query_gpu(min(compute_deadline, hard_deadline - 6.5))
            report["preflight_snapshots"].append(_public_snapshot(snapshot))
            _require_idle(snapshot)
            if expected_uuid is not None and snapshot["_uuid"] != expected_uuid:
                raise ValidationError("GPU identity changed across preflight snapshots")
            expected_uuid = snapshot["_uuid"]
        if _remaining(compute_deadline, 1.0) <= 0:
            raise ValidationError("worker budget exhausted during preflight")
        preflight_passed = True
        report["status"] = "failed"
        buffers = []
        for index in range(WORKER_COUNT):
            process = _spawn_worker()
            processes.append(process)  # retain immediately, before any IPC or second spawn
            buffers.append(bytearray())
            _send_json(process, {"kind": "init", "config": asdict(config),
                                "worker_index": index, "expected_uuid": expected_uuid}, compute_deadline)
        ready = _receive_stage(processes, buffers, "ready", compute_deadline)
        for index, message in enumerate(ready):
            if (message.get("worker_id") != f"owned-worker-{index}" or message.get("status") != "ok"
                    or message.get("ordinal_scope") != "single_device_unset_visibility"):
                raise ValidationError("owned worker readiness/device scope mismatch")
        for process in processes:
            _send_json(process, {"kind": "go"}, compute_deadline)
        results = _receive_stage(processes, buffers, "final", compute_deadline)
        report["workers"] = results
        for index, result in enumerate(results):
            _validate_worker_report(result, index, config)
        peak_reserved = sum(result["peak_reserved_bytes"] for result in results)
        peak_allocated = sum(result["peak_allocated_bytes"] for result in results)
        report["resource_budget"]["owned_cuda_peak_reserved_bytes_sum"] = peak_reserved
        report["resource_budget"]["owned_cuda_peak_allocated_bytes_sum"] = peak_allocated
        report["resource_budget"]["cpu_buffers_plus_owned_cuda_peak_reserved_bytes"] = config.elements * 48 + peak_reserved
        if config.elements * 48 + peak_reserved > MAX_TENSOR_BYTES:
            raise ValidationError("CPU buffers plus owned allocator peak reservation exceed 1 GiB")
        if not _wait_exits(processes, compute_deadline):
            raise ValidationError("owned workers reported but did not exit before deadline")
        if any(process.returncode != 0 for process in processes):
            raise ValidationError("owned worker nonzero exit")
        report["host_launch_overlap"] = _host_overlap(results)
        report["status"] = "ok"
    except Exception as exc:
        # Exceptions are diagnostic; no unbounded traceback or device identity persisted.
        report["diagnostics"].append({"stage": "workers" if preflight_passed else "preflight",
                                      "error": f"{type(exc).__name__}: {str(exc)[:180]}"})
        report["status"] = "failed" if preflight_passed else "blocked"
    finally:
        if preflight_passed:
            cleanup_deadline = min(hard_deadline - 2.5, time.monotonic() + 4.0)
            states = _cleanup_workers(processes, cleanup_deadline)
            report["cleanup"]["owned_processes"] = states
            post = None
            try:
                post = _query_gpu(hard_deadline - 0.5)
                report["cleanup"]["post_gpu_aggregate"] = _public_snapshot(post)
                _require_idle(post)
                if post["_uuid"] != expected_uuid:
                    raise ValidationError("GPU identity changed after cleanup")
            except Exception as exc:
                report["diagnostics"].append({"stage": "cleanup",
                                              "error": f"{type(exc).__name__}: {str(exc)[:180]}"})
                post = None
            allocators_clean = len(results) == WORKER_COUNT and all(
                isinstance(result.get("cleanup"), dict)
                and result["cleanup"].get("completed") is True
                and type(result["cleanup"].get("allocated_bytes_after")) is int
                and result["cleanup"]["allocated_bytes_after"] == 0
                and type(result["cleanup"].get("reserved_bytes_after")) is int
                and result["cleanup"]["reserved_bytes_after"] == 0 for result in results)
            verified = (len(states) == WORKER_COUNT and all(state["confirmed_dead"] for state in states)
                        and allocators_clean and post is not None)
            report["cleanup"]["state"] = "verified" if verified else "unknown"
            if not verified:
                report["status"] = "failed"
        for process in processes:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError as exc:
                        report["status"] = "failed"
                        report["diagnostics"].append({"stage": "stream_cleanup", "error": type(exc).__name__})
        elapsed = time.monotonic() - entered
        report["wall_elapsed_seconds"] = _finite_number(elapsed, "wall elapsed")
        report["wall_measurement_scope"] = "through_cleanup_before_report_persistence"
        report["wall_deadline_met"] = elapsed <= WALL_LIMIT_SECONDS
        if not report["wall_deadline_met"]:
            report["status"] = "failed"
            report["diagnostics"].append({"stage": "deadline", "error": "30 second wall deadline exceeded"})
        descriptor = os.open(destination / "report.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        after_write = time.monotonic() - entered
        report["through_report_write_wall_elapsed_seconds"] = _finite_number(after_write, "wall elapsed after report write")
        report["function_deadline_met_after_report_write"] = after_write <= WALL_LIMIT_SECONDS
        if not report["function_deadline_met_after_report_write"]:
            report["status"] = "failed"
            report["wall_deadline_met"] = False
            report["diagnostics"].append({"stage": "report_persistence", "error": "30 second deadline exceeded during diagnostic write"})
            # This file was exclusively created by this call. Persist a failure if
            # filesystem latency breached the deadline; never claim real-time OS I/O.
            with (destination / "report.json").open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
    return report


def _emit_worker(value: dict[str, Any]) -> None:
    frame = json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n"
    if len(frame.encode("utf-8")) > MAX_IPC_BYTES:
        raise ValidationError("owned worker report exceeds primitive IPC budget")
    sys.stdout.write(frame)
    sys.stdout.flush()


def _read_worker_command() -> dict[str, Any]:
    frame = sys.stdin.buffer.readline(MAX_IPC_BYTES + 1)
    if not frame.endswith(b"\n") or len(frame) > MAX_IPC_BYTES:
        raise ValidationError("owned worker command is missing or oversized")
    value = json.loads(frame)
    if not isinstance(value, dict):
        raise ValidationError("owned worker command must be an object")
    return value


def _owned_worker() -> int:
    """The only CUDA import/initialization path, invoked by the owned exec-spawn."""
    index = -1
    torch = None
    gpu_input = gpu_output = cpu_input = cpu_oracle = cpu_copy = None
    start_event = end_event = None
    final: dict[str, Any] = {"kind": "final", "status": "failed", "steps": [],
        "empty_event_pairs": [], "ground_truth_all_passed": False, "cleanup": {"completed": False}}
    try:
        command = _read_worker_command()
        if command.get("kind") != "init" or type(command.get("worker_index")) is not int or command["worker_index"] not in (0, 1):
            raise ValidationError("owned worker initialization contract invalid")
        index = command["worker_index"]
        config = OwnedGPUTimingConfig(**command["config"])
        config.validate()
        expected_uuid = _normalize_uuid(command.get("expected_uuid"))
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            raise ValidationError("owned worker CUDA visibility became ambiguous")
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(1)  # only this owned process; no environment/configuration mutation
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValidationError("owned worker requires exactly one visible CUDA GPU")
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        uuid_verified = _verify_worker_uuid(properties, expected_uuid)
        torch.cuda.reset_peak_memory_stats(0)
        gpu_input = torch.empty(config.elements, dtype=torch.int64, device="cuda:0")
        gpu_output = torch.empty(config.elements, dtype=torch.int64, device="cuda:0")
        cpu_input = torch.arange(config.elements, dtype=torch.int64, device="cpu")
        cpu_oracle = torch.empty(config.elements, dtype=torch.int64, device="cpu")
        cpu_copy = torch.empty(config.elements, dtype=torch.int64, device="cpu")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        final.update(worker_id=f"owned-worker-{index}", elements=config.elements, tensor_dtype="int64",
                     device_uuid_crosschecked_when_available=uuid_verified)
        _emit_worker({"kind": "ready", "worker_id": f"owned-worker-{index}", "status": "ok",
                      "ordinal_scope": "single_device_unset_visibility"})
        if _read_worker_command() != {"kind": "go"}:
            raise ValidationError("owned worker start command invalid")
        previous_start = None
        for pair in range(3):
            before = time.monotonic_ns()
            start_event.record()
            end_event.record()
            after = time.monotonic_ns()
            end_event.synchronize()
            synchronized = time.monotonic_ns()
            elapsed = _finite_number(start_event.elapsed_time(end_event), "empty-event elapsed")
            row = {"host_before_start_record_ns": before, "host_after_end_record_ns": after,
                   "host_after_synchronize_ns": synchronized, "event_elapsed_ms": elapsed,
                   "host_completion_bracket_ns": synchronized - before,
                   "host_minus_event_slack_ns": synchronized - before - elapsed * 1e6,
                   "host_start_spacing_ns": None if previous_start is None else before - previous_start}
            final["empty_event_pairs"].append(row)
            previous_start = before
        for step in range(config.steps):
            before = time.monotonic_ns()
            start_event.record()
            torch.arange(config.elements, out=gpu_input)
            torch.add(gpu_input, step + 1, out=gpu_output)
            end_event.record()
            after = time.monotonic_ns()
            end_event.synchronize()
            synchronized = time.monotonic_ns()
            elapsed = _finite_number(start_event.elapsed_time(end_event), "operation elapsed")
            # Preallocated full copy prevents transient sixth-vector peaks.
            cpu_copy.copy_(gpu_output)
            torch.add(cpu_input, step + 1, out=cpu_oracle)
            exact = (gpu_input.dtype == torch.int64 and gpu_output.dtype == torch.int64
                     and cpu_copy.dtype == torch.int64 and bool(torch.equal(cpu_copy, cpu_oracle)))
            if not exact:
                raise ValidationError("exact full-vector CPU/GPU int64 ground truth mismatch")
            final["steps"].append({"sample_id": f"owned-worker-{index}-sample-{step}",
                "step_id": f"owned-worker-{index}-step-{step}", "step_index": step,
                "host_before_start_record_ns": before, "host_after_end_record_ns": after,
                "host_after_synchronize_ns": synchronized, "event_elapsed_ms": elapsed,
                "host_completion_bracket_ns": synchronized - before,
                "host_minus_event_slack_ns": synchronized - before - elapsed * 1e6,
                "exact_full_vector_equal": exact,
                "nonfinite_values": "impossible_for_int64_and_exact_bounded_oracle"})
        final["selected_operation_event_elapsed_ms_sum"] = sum(row["event_elapsed_ms"] for row in final["steps"])
        final["ground_truth_all_passed"] = True
        final["status"] = "ok"
    except Exception as exc:
        final["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    finally:
        if torch is not None:
            try:
                final["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
                final["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
            except Exception as exc:
                final["status"] = "failed"
                final["error"] = f"allocator peak unavailable: {type(exc).__name__}"
        gpu_input = gpu_output = cpu_input = cpu_oracle = cpu_copy = None
        start_event = end_event = None
        gc.collect()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
                allocated = int(torch.cuda.memory_allocated(0))
                reserved = int(torch.cuda.memory_reserved(0))
                final["cleanup"] = {"completed": True, "allocated_bytes_after": allocated,
                                     "reserved_bytes_after": reserved}
            except Exception as exc:
                final["cleanup"] = {"completed": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
                final["status"] = "failed"
        final.setdefault("worker_id", f"owned-worker-{index}")
        try:
            _emit_worker(final)
        except (BrokenPipeError, OSError, ValidationError):
            return 1
    return 0 if final["status"] == "ok" else 1


if __name__ == "__main__":
    if sys.argv[1:] != ["--owned-worker"]:
        raise SystemExit("Use the owned-gpu-timing-benchmark CLI; worker mode is internal.")
    raise SystemExit(_owned_worker())
