"""Static KV storage and CPU-testable attention over declared selected positions.

The audit describes logical tensor selection. It cannot verify physical cache
line accesses or turn a Python index into a cache/TLB observation.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, TYPE_CHECKING

from .schema import ValidationError

if TYPE_CHECKING:
    from .selector_replay import SelectorReplayConfig

MAX_CACHE_BYTES = 64 * 1024 ** 2
AUDIT_VERSION = "janus.selector_execution_audit.v1"


def _config_contract(config: Any) -> str:
    from .selector_replay import SelectorReplayConfig
    if not isinstance(config, SelectorReplayConfig):
        raise ValidationError("config must be SelectorReplayConfig")
    config.validate()
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _index(value: Any, limit: int, name: str) -> int:
    if type(value) is not int or not 0 <= value < limit:
        raise ValidationError(f"{name} must be a builtin integer in [0, {limit})")
    return value


def _positions(value: Any, valid_length: int) -> list[int]:
    if type(value) not in (list, tuple) or not value:
        raise ValidationError("positions must be a nonempty list/tuple")
    positions = list(value)
    if any(type(position) is not int or not 0 <= position < valid_length for position in positions):
        raise ValidationError("positions must be builtin integer indices within valid cache rows")
    if any(left >= right for left, right in zip(positions, positions[1:])):
        raise ValidationError("positions must be sorted and unique")
    return positions


def _finite_tensor(torch: Any, tensor: Any, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValidationError(f"{name} must contain finite values")


class StaticKVCache:
    """Two fixed allocations with append-only writes of one position per layer."""

    def __init__(self, config: SelectorReplayConfig, *, dtype: Any = None, device: Any = "cpu"):
        contract = _config_contract(config)
        import torch
        resolved_dtype = torch.float32 if dtype is None else dtype
        if resolved_dtype not in (torch.float32, torch.float64):
            raise ValidationError("static cache supports float32 and float64")
        try:
            resolved_device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValidationError("device must be a valid torch device") from exc
        if resolved_device.type not in ("cpu", "cuda"):
            raise ValidationError("static cache device must be CPU or explicitly requested CUDA")
        capacity = len(config.prompt_token_ids) + len(config.teacher_forced_token_ids) - 1
        itemsize = 4 if resolved_dtype == torch.float32 else 8
        cache_bytes = 2 * config.layers * config.kv_heads * capacity * config.head_dim * itemsize
        if cache_bytes > MAX_CACHE_BYTES:
            raise ValidationError("static KV cache exceeds 64 MiB allocation budget")
        self._config = config
        self._contract = contract
        self.capacity = capacity
        self.dtype = resolved_dtype
        self.cache_bytes = cache_bytes
        shape = (config.layers, config.kv_heads, capacity, config.head_dim)
        self._keys = torch.empty(shape, dtype=resolved_dtype, device=resolved_device)
        self._values = torch.empty(shape, dtype=resolved_dtype, device=resolved_device)
        self.device = self._keys.device
        self._valid_lengths = [0] * config.layers

    @property
    def config(self) -> SelectorReplayConfig:
        return self._config

    @property
    def keys(self) -> Any:
        return self._keys

    @property
    def values(self) -> Any:
        return self._values

    @property
    def valid_lengths(self) -> tuple[int, ...]:
        return tuple(self._valid_lengths)

    def _metadata(self) -> None:
        # Shape/dtype/device only: no reads or scans of unselected tensor values.
        if _config_contract(self.config) != self._contract:
            raise ValidationError("cache config changed after allocation")
        expected_capacity = len(self.config.prompt_token_ids) + len(self.config.teacher_forced_token_ids) - 1
        shape = (self.config.layers, self.config.kv_heads, expected_capacity, self.config.head_dim)
        if self.capacity != expected_capacity:
            raise ValidationError("cache capacity metadata changed")
        for tensor in (self.keys, self.values):
            if tuple(tensor.shape) != shape or tensor.dtype != self.dtype or tensor.device != self.device:
                raise ValidationError("cache storage shape/dtype/device mismatch")
        if len(self._valid_lengths) != self.config.layers or any(
            type(length) is not int or not 0 <= length <= self.capacity for length in self._valid_lengths
        ):
            raise ValidationError("cache valid-length metadata is invalid")

    def _row_tensor(self, tensor: Any, name: str) -> None:
        import torch
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"{name} must be a torch tensor")
        if tuple(tensor.shape) != (self.config.kv_heads, self.config.head_dim):
            raise ValidationError(f"{name} shape must be [kv_heads, head_dim]")
        if tensor.dtype != self.dtype or tensor.device != self.device:
            raise ValidationError(f"{name} dtype/device must match cache")
        _finite_tensor(torch, tensor, name)

    def write(self, layer_index: int, position: int, key: Any, value: Any) -> None:
        import torch
        self._metadata()
        layer = _index(layer_index, self.config.layers, "layer_index")
        pos = _index(position, self.capacity, "position")
        if pos != self._valid_lengths[layer]:
            raise ValidationError("cache writes must append exactly the next position; overwrite/gap rejected")
        # Validate both complete incoming rows before mutating either allocation.
        self._row_tensor(key, "key")
        self._row_tensor(value, "value")
        with torch.no_grad():
            self.keys[layer, :, pos, :].copy_(key)
            self.values[layer, :, pos, :].copy_(value)
        self._valid_lengths[layer] += 1

    def gather(self, layer_index: int, kv_head_index: int, positions: list[int]) -> tuple[Any, Any]:
        import torch
        self._metadata()
        layer = _index(layer_index, self.config.layers, "layer_index")
        head = _index(kv_head_index, self.config.kv_heads, "kv_head_index")
        selected = _positions(positions, self._valid_lengths[layer])
        indices = torch.tensor(selected, dtype=torch.long, device=self.device)
        # Only the requested head/positions are materialized, never a full cache copy.
        keys = torch.index_select(self.keys[layer, head], 0, indices)
        values = torch.index_select(self.values[layer, head], 0, indices)
        _finite_tensor(torch, keys, "selected keys")
        _finite_tensor(torch, values, "selected values")
        return keys, values


def selected_attention(query: Any, cache: StaticKVCache, manifest: dict[str, Any], *,
                       step_index: int, layer_index: int,
                       expected_config: SelectorReplayConfig | None = None) -> tuple[Any, dict[str, Any]]:
    """GQA with one union gather per KV head, then per-query-head subsets."""
    from .selector_replay import validate_selector_manifest
    import torch
    if not isinstance(cache, StaticKVCache):
        raise ValidationError("cache must be StaticKVCache")
    cache._metadata()
    if expected_config is not None and _config_contract(expected_config) != cache._contract:
        raise ValidationError("expected_config must match the allocated cache contract")
    canonical = validate_selector_manifest(manifest, expected_config=cache.config)
    step = _index(step_index, len(cache.config.teacher_forced_token_ids), "step_index")
    layer = _index(layer_index, cache.config.layers, "layer_index")
    valid_length = len(cache.config.prompt_token_ids) + step
    if cache.valid_lengths[layer] != valid_length:
        raise ValidationError("cache valid length must equal prompt length plus step; missing/future rows rejected")
    if not isinstance(query, torch.Tensor) or tuple(query.shape) != (cache.config.query_heads, cache.config.head_dim):
        raise ValidationError("query must have shape [query_heads, head_dim]")
    if query.dtype != cache.dtype or query.device != cache.device:
        raise ValidationError("query dtype/device must match cache")
    _finite_tensor(torch, query, "query")
    selections = [entry for entry in canonical["steps"][step]["selections"] if entry["layer_index"] == layer]
    if [entry["query_head_index"] for entry in selections] != list(range(cache.config.query_heads)):
        raise ValidationError("manifest must supply exactly one canonical selection per query head")
    query_heads_per_kv = cache.config.query_heads // cache.config.kv_heads
    if query_heads_per_kv < 1 or cache.config.query_heads % cache.config.kv_heads:
        raise ValidationError("query heads must be an exact multiple of KV heads")
    output = torch.empty_like(query)
    groups = []
    for kv_head in range(cache.config.kv_heads):
        head_selections = selections[kv_head * query_heads_per_kv:(kv_head + 1) * query_heads_per_kv]
        if any(entry["kv_head_index"] != kv_head for entry in head_selections):
            raise ValidationError("manifest GQA mapping differs from q // queries_per_KV")
        union_positions = sorted({position for entry in head_selections for position in entry["absolute_kv_positions"]})
        selected_keys, selected_values = cache.gather(layer, kv_head, union_positions)
        union_offsets = {position: offset for offset, position in enumerate(union_positions)}
        per_query = []
        for entry in head_selections:
            head = entry["query_head_index"]
            positions = _positions(entry["absolute_kv_positions"], valid_length)
            offsets = torch.tensor([union_offsets[position] for position in positions], dtype=torch.long, device=cache.device)
            # These second selections read only the small union buffers, not the cache.
            keys = torch.index_select(selected_keys, 0, offsets)
            values = torch.index_select(selected_values, 0, offsets)
            scores = torch.matmul(keys, query[head]) / (cache.config.head_dim ** 0.5)
            _finite_tensor(torch, scores, "selected logits")
            probabilities = torch.softmax(scores, dim=0)
            _finite_tensor(torch, probabilities, "selected probabilities")
            result = torch.matmul(probabilities, values)
            _finite_tensor(torch, result, "selected output")
            output[head] = result
            per_query.append({"query_head_index": head, "absolute_kv_positions": positions,
                              "selected_row_count": len(positions)})
        groups.append({"kv_head_index": kv_head,
            "union_selected_positions": union_positions, "union_selected_row_count": len(union_positions),
            "source_allocated_row_count": cache.capacity, "source_valid_row_count": valid_length,
            "source_cache_bytes_theoretical": 2 * cache.capacity * cache.config.head_dim * output.element_size(),
            "source_valid_prefix_bytes_theoretical": 2 * valid_length * cache.config.head_dim * output.element_size(),
            "gathered_bytes_theoretical": 2 * len(union_positions) * cache.config.head_dim * output.element_size(),
            "source_index_select_calls": {"keys": 1, "values": 1}, "query_heads": per_query})
    _finite_tensor(torch, output, "attention output")
    audit = {"schema_version": AUDIT_VERSION, "source_kind": "victim_selector_replay_execution",
        "evidence_scope": "private_logical_tensor_selection_only", "physical_cacheline_access_verified": False,
        "physical_gpu_memory_traffic_verified": False, "is_probe_observation": False,
        "step_index": step, "step_id": canonical["steps"][step]["step_id"], "layer_index": layer,
        "query_position": valid_length - 1, "cache_valid_length": valid_length,
        "cache_capacity": cache.capacity, "dtype": str(cache.dtype), "device": str(cache.device),
        "query_heads_per_kv_head": query_heads_per_kv, "kv_head_groups": groups,
        "source_cache_bytes_theoretical": sum(group["source_cache_bytes_theoretical"] for group in groups),
        "source_valid_prefix_bytes_theoretical": sum(group["source_valid_prefix_bytes_theoretical"] for group in groups),
        "gathered_bytes_theoretical": sum(group["gathered_bytes_theoretical"] for group in groups),
        "byte_count_interpretation": "logical K/V row storage only, not measured hardware traffic"}
    return output, audit
