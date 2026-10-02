"""Strict metadata boundary for a future owned-buffer probe implementation.

This module only validates host scheduling and bounded buffer descriptors. It
allocates no buffers, executes no probes, and accepts no measured observations
or GPU clock. Opaque ID syntax cannot prove that an ID lacks semantic content
or that a buffer is owned. This is metadata separation, not an OS sandbox.
"""

from __future__ import annotations

import re
from typing import Any

from .schema import ValidationError


SCHEMA_VERSION = "janus.probe.request.v1"
MAX_HOST_TIME_NS = 2**63 - 1
MAX_BUFFERS = 4
MAX_BUFFER_ELEMENTS = 1_048_576
MAX_OWN_BUFFER_BYTES = 64 * 1024 * 1024
_DTYPE_BYTES = {"int64": 8, "float32": 4}
_ID = re.compile(r"[0-9a-f]{32}")


def _exact_keys(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValidationError(f"{name} must be a builtin object")
    if any(type(key) is not str for key in value) or set(value) != fields:
        raise ValidationError(f"{name} fields must equal {sorted(fields)}")
    return value


def _opaque_id(value: Any, name: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValidationError(f"{name} must be a 32-character lowercase hexadecimal ID")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError(f"{name} must be a builtin integer in [{minimum}, {maximum}]")
    return value


def validate_probe_request(request: dict[str, Any]) -> dict[str, Any]:
    """Return a fresh canonical primitive request after strict v1 validation.

    The timing window describes host scheduler bounds only. It is not a probe
    observation, latency measurement, or claim that a probe was executed.
    Validation precedes copying; output contains only the exact v1 fields.
    """
    request = _exact_keys(
        request, {"schema_version", "run_id", "step_id", "timing", "buffers"}, "request"
    )
    version = request["schema_version"]
    if type(version) is not str or version != SCHEMA_VERSION:
        raise ValidationError(f"schema_version must equal {SCHEMA_VERSION}")
    run_id = _opaque_id(request["run_id"], "run_id")
    step_id = _opaque_id(request["step_id"], "step_id")
    timing = _exact_keys(
        request["timing"], {"host_not_before_ns", "host_deadline_ns"}, "timing"
    )
    start = _integer(
        timing["host_not_before_ns"], "host_not_before_ns", 0, MAX_HOST_TIME_NS
    )
    deadline = _integer(
        timing["host_deadline_ns"], "host_deadline_ns", 0, MAX_HOST_TIME_NS
    )
    if deadline <= start:
        raise ValidationError("host_deadline_ns must be greater than host_not_before_ns")

    buffers = request["buffers"]
    if type(buffers) is not list or not 1 <= len(buffers) <= MAX_BUFFERS:
        raise ValidationError(f"buffers must be a builtin array containing 1..{MAX_BUFFERS} entries")
    seen: set[str] = set()
    descriptors = []
    total_bytes = 0
    for index, buffer in enumerate(buffers):
        name = f"buffers[{index}]"
        buffer = _exact_keys(buffer, {"buffer_id", "elements", "dtype"}, name)
        buffer_id = _opaque_id(buffer["buffer_id"], f"{name}.buffer_id")
        if buffer_id in seen:
            raise ValidationError("duplicate buffer_id")
        seen.add(buffer_id)
        elements = _integer(
            buffer["elements"], f"{name}.elements", 1, MAX_BUFFER_ELEMENTS
        )
        dtype = buffer["dtype"]
        if type(dtype) is not str or dtype not in _DTYPE_BYTES:
            raise ValidationError(f"{name}.dtype must equal int64 or float32")
        total_bytes += elements * _DTYPE_BYTES[dtype]
        if total_bytes > MAX_OWN_BUFFER_BYTES:
            raise ValidationError("total owned-buffer byte budget exceeded")
        descriptors.append({"buffer_id": buffer_id, "elements": elements, "dtype": dtype})

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "step_id": step_id,
        "timing": {"host_not_before_ns": start, "host_deadline_ns": deadline},
        "buffers": descriptors,
    }
