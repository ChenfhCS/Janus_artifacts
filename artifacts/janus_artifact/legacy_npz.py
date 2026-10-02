"""Conservative adapter for legacy reconstructed prefill rank tensors."""

from __future__ import annotations

import ast
import hashlib
import struct
import zipfile
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION, ValidationError, validate_trace_record


EXPECTED_KEYS = {"attn_rank", "top_k"}
DEFAULT_MAX_FILE_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_NPY_HEADER_BYTES = 10_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_npy_header(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    max_uncompressed_bytes: int,
) -> None:
    """Bound the declared array before NumPy can allocate its data buffer."""
    import numpy as np

    with archive.open(member) as handle:
        prefix = handle.read(8)
        if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
            raise ValidationError("legacy NPZ member is not a valid NPY array")
        version = tuple(prefix[6:])
        if version == (1, 0):
            length_size, length_format, encoding = 2, "<H", "latin1"
        elif version in ((2, 0), (3, 0)):
            length_size, length_format = 4, "<I"
            encoding = "utf8" if version == (3, 0) else "latin1"
        else:
            raise ValidationError("legacy NPZ contains an unsupported NPY version")
        length_data = handle.read(length_size)
        if len(length_data) != length_size:
            raise ValidationError("legacy NPZ contains a truncated NPY header")
        header_size = struct.unpack(length_format, length_data)[0]
        if not 0 < header_size <= MAX_NPY_HEADER_BYTES:
            raise ValidationError("legacy NPZ exceeds the NPY header-size limit")
        data_offset = 8 + length_size + header_size
        if data_offset > member.file_size:
            raise ValidationError("legacy NPZ contains a truncated NPY header")
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValidationError("legacy NPZ contains a truncated NPY header")
        try:
            header = ast.literal_eval(header_bytes.decode(encoding))
        except (ValueError, SyntaxError, UnicodeError, RecursionError) as exc:
            raise ValidationError("legacy NPZ contains an invalid NPY header") from exc

    if not isinstance(header, dict) or set(header) != {
        "descr", "fortran_order", "shape"
    }:
        raise ValidationError("legacy NPZ contains an invalid NPY header")
    if type(header["fortran_order"]) is not bool:
        raise ValidationError("legacy NPZ contains an invalid NPY storage order")
    shape = header["shape"]
    if not isinstance(shape, tuple) or any(type(dimension) is not int for dimension in shape):
        raise ValidationError("legacy NPZ contains an invalid NPY shape")
    # The adapter only accepts plain integer tensors. Reject structured and
    # subarray descriptors before dtype construction as well as object buffers.
    if not isinstance(header["descr"], str):
        raise ValidationError("legacy NPZ arrays must have a plain integer dtype")
    try:
        dtype = np.dtype(header["descr"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("legacy NPZ contains an invalid NPY dtype") from exc
    if dtype.hasobject:
        raise ValidationError("object dtype is forbidden in legacy NPZ")
    if not np.issubdtype(dtype, np.integer):
        raise ValidationError("legacy NPZ arrays must have an integer dtype")
    if member.filename == "attn_rank.npy":
        if len(shape) != 4:
            raise ValidationError("attn_rank must be a four-dimensional integer tensor")
        if any(dimension <= 0 for dimension in shape):
            raise ValidationError("attn_rank dimensions must be positive")
    elif shape != ():
        raise ValidationError("top_k must be an integer scalar")

    # Python integers do not wrap; short-circuit before multiplying huge
    # dimensions. Each array must fit the already checked archive budget.
    declared_bytes = int(dtype.itemsize)
    for dimension in shape:
        if dimension > max_uncompressed_bytes // declared_bytes:
            raise ValidationError("legacy NPZ declared array exceeds the uncompressed-size limit")
        declared_bytes *= dimension
    if declared_bytes > max_uncompressed_bytes:
        raise ValidationError("legacy NPZ declared array exceeds the uncompressed-size limit")
    if declared_bytes != member.file_size - data_offset:
        raise ValidationError("legacy NPZ NPY declared array bytes do not match the member body")


def _inspect_zip(path: Path, max_uncompressed_bytes: int) -> list[dict[str, Any]]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if not members:
                raise ValidationError("legacy NPZ archive is empty")
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise ValidationError("legacy NPZ contains duplicate member names")
            if any(
                member.is_dir()
                or "/" in member.filename
                or "\\" in member.filename
                or not member.filename.endswith(".npy")
                for member in members
            ):
                raise ValidationError("legacy NPZ contains an unexpected member path or type")
            if set(names) != {f"{key}.npy" for key in EXPECTED_KEYS}:
                raise ValidationError(
                    f"legacy prefill NPZ keys must equal {sorted(EXPECTED_KEYS)}"
                )
            total_uncompressed = 0
            for member in members:
                if member.file_size < 0 or member.compress_size < 0:
                    raise ValidationError("legacy NPZ contains an invalid member size")
                if member.file_size > max_uncompressed_bytes:
                    raise ValidationError("legacy NPZ exceeds the uncompressed-size limit")
                total_uncompressed += member.file_size
                if total_uncompressed > max_uncompressed_bytes:
                    raise ValidationError("legacy NPZ exceeds the uncompressed-size limit")
            # No member is decompressed until directory names and sizes pass.
            # Only bounded headers are read before any array allocation or CRC
            # traversal of the complete members.
            for member in members:
                _inspect_npy_header(archive, member, max_uncompressed_bytes)
            if archive.testzip() is not None:
                raise ValidationError("NPZ CRC verification failed")
    except zipfile.BadZipFile as exc:
        raise ValidationError("legacy payload is not a valid NPZ ZIP archive") from exc
    except (RuntimeError, NotImplementedError, EOFError, OSError) as exc:
        raise ValidationError("legacy NPZ member cannot be read safely") from exc
    return [
        {
            "name": member.filename,
            "compressed_bytes": member.compress_size,
            "uncompressed_bytes": member.file_size,
        }
        for member in members
    ]


def inspect_prefill_rank_npz(
    payload_path: str | Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> dict[str, Any]:
    """Inspect two declared numeric arrays with NumPy pickle loading disabled."""
    path = Path(payload_path)
    if not path.is_file():
        raise ValidationError(f"legacy NPZ does not exist: {path}")
    size_bytes = path.stat().st_size
    if size_bytes <= 0 or size_bytes > max_file_bytes:
        raise ValidationError("legacy NPZ exceeds the file-size limit")
    try:
        import numpy as np
    except ImportError as exc:
        raise ValidationError("NumPy is required for safe legacy NPZ inspection") from exc
    zip_members = _inspect_zip(path, max_uncompressed_bytes)

    arrays: list[dict[str, Any]] = []
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != EXPECTED_KEYS:
                raise ValidationError(
                    f"legacy prefill NPZ keys must equal {sorted(EXPECTED_KEYS)}"
                )
            attn_rank = payload["attn_rank"]
            top_k = payload["top_k"]
            if attn_rank.dtype.hasobject or top_k.dtype.hasobject:
                raise ValidationError("object dtype is forbidden in legacy NPZ")
            if attn_rank.ndim != 4 or not np.issubdtype(attn_rank.dtype, np.integer):
                raise ValidationError("attn_rank must be a four-dimensional integer tensor")
            if any(dimension <= 0 for dimension in attn_rank.shape):
                raise ValidationError("attn_rank dimensions must be positive")
            if top_k.ndim != 0 or not np.issubdtype(top_k.dtype, np.integer):
                raise ValidationError("top_k must be an integer scalar")
            top_k_value = int(top_k)
            if top_k_value <= 0:
                raise ValidationError("top_k must be positive")
            for key, value in (("attn_rank", attn_rank), ("top_k", top_k)):
                contiguous = np.ascontiguousarray(value)
                arrays.append(
                    {
                        "key": key,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "nbytes": int(value.nbytes),
                        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
                        "min": int(value.min()) if value.size else None,
                        "max": int(value.max()) if value.size else None,
                    }
                )
    except ValidationError:
        raise
    except ValueError as exc:
        raise ValidationError(
            "legacy NPZ contains an unsupported or pickle-backed array"
        ) from exc

    return {
        "sha256": _sha256_file(path),
        "size_bytes": size_bytes,
        "zip_members": zip_members,
        "arrays": arrays,
    }


def adapt_prefill_rank_npz(
    payload_path: str | Path, *, legacy_path: str, case_id: str
) -> dict[str, Any]:
    if not isinstance(legacy_path, str) or not legacy_path:
        raise ValidationError("legacy_path must be a non-empty string")
    if not isinstance(case_id, str) or not case_id:
        raise ValidationError("case_id must be a non-empty string supplied by the caller")
    inspection = inspect_prefill_rank_npz(payload_path)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": f"legacy-prefill-rank-{inspection['sha256'][:16]}",
        "case_id": case_id,
        "split": "unassigned",
        "phase": "prefill",
        "source_kind": "refactored_legacy",
        "payload_status": "materialized_verified",
        "trace": {
            "data_stage": "reconstructed_sparsity",
            "granularity": "token",
            "aggregation": "rank_tensor",
            "storage": "external_npz",
            "external_payload": {
                "legacy_path": legacy_path,
                **inspection,
            },
        },
        "labels": {},
        "evaluation_eligibility": {
            "pasr": False,
            "dasr": False,
            "reasons": [
                "split is unassigned",
                "no explicit ground-truth join contract is provided",
                "no prediction record is associated by stable ID",
                "this is a reconstructed prefill tensor, not raw probe data",
            ],
        },
        "provenance": {
            "legacy_path": legacy_path,
            "adaptation": "safe_npz_allow_pickle_false",
            "raw_probe_trace": False,
        },
    }
    validate_trace_record(record)
    return record


def verify_prefill_rank_npz_record(
    record: dict[str, Any], payload_path: str | Path
) -> None:
    validate_trace_record(record)
    trace = record.get("trace", {})
    if trace.get("storage") != "external_npz":
        raise ValidationError("record does not reference an external NPZ")
    expected = trace.get("external_payload")
    actual = inspect_prefill_rank_npz(payload_path)
    for field in ("sha256", "size_bytes", "zip_members", "arrays"):
        if not isinstance(expected, dict) or expected.get(field) != actual[field]:
            raise ValidationError(f"external NPZ {field} mismatch")
