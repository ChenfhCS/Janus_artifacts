"""CPU-only selector/cache/attention integration with an isolated oracle sidecar."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

from .schema import ValidationError
from .selector_replay import (
    SelectorReplayConfig, build_selector_manifest, validate_selector_manifest,
)
from .probe_boundary import validate_probe_request
from .sparse_attention import StaticKVCache, selected_attention


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_private_json(destination: Path, value: dict) -> None:
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)


def synthetic_selector_fixture():
    """Return explicit CPU inputs; this is not a pretrained model forward pass."""
    import torch

    dimensions = {"layers": 2, "query_heads": 4, "kv_heads": 2, "head_dim": 3}
    config = SelectorReplayConfig(
        run_id=_digest("synthetic-selector-replay-v1")[:32],
        model_id="synthetic-selector-replay",
        model_revision="fixture-v1",
        tokenizer_id="synthetic-integer-tokenizer",
        tokenizer_revision="fixture-v1",
        model_config_sha256=_digest(dimensions),
        weights_sha256=None,
        source_kind="synthetic_tensor_fixture",
        seed=41,
        prompt_token_ids=(7, 11, 13, 17),
        teacher_forced_token_ids=(19, 23, 29),
        top_k=1,
        include_latest=True,
        **dimensions,
    )
    config.validate()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    cache_shape = (config.layers, config.kv_heads, config.cache_capacity, config.head_dim)
    keys = torch.randn(cache_shape, dtype=torch.float64, device="cpu", generator=generator)
    values = torch.randn(cache_shape, dtype=torch.float64, device="cpu", generator=generator)
    queries = torch.randn(
        (len(config.teacher_forced_token_ids), config.layers, config.query_heads, config.head_dim),
        dtype=torch.float64, device="cpu", generator=generator,
    )
    # Each initialized cache row is associated with this exact causal token history.
    token_history = config.prompt_token_ids + config.teacher_forced_token_ids[:-1]
    token_offsets = torch.tensor(token_history, dtype=torch.float64, device="cpu") * 0.001
    keys.add_(token_offsets[None, None, :, None])
    values.add_(token_offsets[None, None, :, None])
    score_rows = []
    for step in range(len(config.teacher_forced_token_ids)):
        length = len(config.prompt_token_ids) + step
        for layer in range(config.layers):
            for query_head in range(config.query_heads):
                kv_head = query_head // (config.query_heads // config.kv_heads)
                logits = keys[layer, kv_head, :length] @ queries[step, layer, query_head]
                scores = torch.softmax(logits / math.sqrt(config.head_dim), dim=0).tolist()
                score_rows.append({
                    "step_index": step, "layer_index": layer,
                    "query_head_index": query_head, "scores": scores,
                })
    return config, score_rows, keys, values, queries


def _mathematical_reference(query, keys, values, step, layer, manifest):
    """Evaluation-only reference over selected CPU fixture rows."""
    import torch

    rows = manifest["steps"][step]["selections"]
    outputs = []
    for head in range(query.shape[0]):
        selection = next(row for row in rows if row["layer_index"] == layer and row["query_head_index"] == head)
        positions = selection["absolute_kv_positions"]
        kv_head = selection["kv_head_index"]
        selected_keys = torch.stack([keys[layer, kv_head, p] for p in positions])
        selected_values = torch.stack([values[layer, kv_head, p] for p in positions])
        logits = selected_keys @ query[head] / math.sqrt(query.shape[-1])
        outputs.append(torch.softmax(logits, dim=0) @ selected_values)
    return torch.stack(outputs)


def run_selector_replay_smoke(work_dir: str | Path) -> dict:
    """Run the complete synthetic CPU path; no model or probe is executed."""
    import torch

    destination = Path(work_dir)
    try:
        destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValidationError("work_dir must be new; existing files are preserved") from exc
    entered = time.monotonic()
    config, score_rows, keys, values, queries = synthetic_selector_fixture()
    manifest = build_selector_manifest(config, score_rows)
    validate_selector_manifest(manifest, expected_config=config)
    # Freeze the manifest to disk before starting the replay cache.
    _write_private_json(destination / "selector-manifest.json", manifest)
    _write_private_json(destination / "offline-scores.json", {"source_kind": config.source_kind, "rows": score_rows})
    cache = StaticKVCache(config, dtype=torch.float64, device="cpu")
    storage_before = [cache.keys.data_ptr(), cache.values.data_ptr()]
    for position in range(len(config.prompt_token_ids)):
        for layer in range(config.layers):
            cache.write(layer, position, keys[layer, :, position], values[layer, :, position])
    evaluations = []
    for step in range(len(config.teacher_forced_token_ids)):
        if step:
            position = len(config.prompt_token_ids) + step - 1
            for layer in range(config.layers):
                cache.write(layer, position, keys[layer, :, position], values[layer, :, position])
        for layer in range(config.layers):
            output, audit = selected_attention(
                queries[step, layer], cache, manifest,
                step_index=step, layer_index=layer, expected_config=config,
            )
            expected = _mathematical_reference(queries[step, layer], keys, values, step, layer, manifest)
            torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
            if not torch.isfinite(output).all().item():
                raise ValidationError("synthetic selected attention output is nonfinite")
            evaluations.append({
                "step_id": manifest["steps"][step]["step_id"], "layer_index": layer,
                "output_shape": list(output.shape), "reference_close": True,
                "max_absolute_error": float((output - expected).abs().max()),
                "logical_access_audit": audit,
            })
    storage_after = [cache.keys.data_ptr(), cache.values.data_ptr()]
    if storage_before != storage_after:
        raise ValidationError("static cache storage changed during replay")
    # This is an input-contract check only. No buffers or latency observations are produced.
    schedule_start = time.monotonic_ns()
    requests = []
    for step in manifest["steps"]:
        request = {
            "schema_version": "janus.probe.request.v1", "run_id": config.run_id,
            "step_id": step["step_id"],
            "timing": {"host_not_before_ns": schedule_start, "host_deadline_ns": schedule_start + 1_000_000_000},
            "buffers": [{"buffer_id": _digest("synthetic-probe-buffer-spec")[:32], "elements": 64, "dtype": "int64"}],
        }
        requests.append(validate_probe_request(request))
    _write_private_json(destination / "probe-request-contracts.json", {
        "probe_executed": False, "contract_checks_only": True, "requests": requests,
    })
    _write_private_json(destination / "oracle-sidecar.json", {
        "schema_version": "janus.selector.smoke.oracle.v1", "source_kind": config.source_kind,
        "run_id": config.run_id, "manifest_sha256": manifest["manifest_sha256"],
        "sequence_sha256": manifest["sequence_sha256"],
        "prompt_token_ids": list(config.prompt_token_ids),
        "teacher_forced_token_ids": list(config.teacher_forced_token_ids),
        "evaluations": evaluations,
    })
    report = {
        "status": "ok", "source_kind": config.source_kind, "device": "cpu",
        "scientific_result": False, "paper_reproduction": False,
        "real_model_loaded": False, "probe_executed": False,
        "physical_cacheline_access_verified": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "generation_steps": len(config.teacher_forced_token_ids),
        "layer_step_evaluations": len(evaluations),
        "query_head_evaluations": len(evaluations) * config.query_heads,
        "all_references_close": True, "reference_rtol": 1e-12, "reference_atol": 1e-12,
        "all_gathers_strict_subset_of_valid_prefix": all(
            group["union_selected_row_count"] < group["source_valid_row_count"]
            for row in evaluations for group in row["logical_access_audit"]["kv_head_groups"]
        ),
        "max_absolute_error": max(row["max_absolute_error"] for row in evaluations),
        "static_cache_storage_unchanged": True,
        "static_cache_tensor_bytes": 2 * config.layers * config.kv_heads * config.cache_capacity * config.head_dim * 8,
        "probe_request_contracts_validated": len(requests),
        "oracle_separate_from_probe_requests": True,
        "duration_seconds": time.monotonic() - entered,
    }
    _write_private_json(destination / "report.json", report)
    return report
