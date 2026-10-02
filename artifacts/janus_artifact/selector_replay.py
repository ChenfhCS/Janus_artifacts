"""Bounded CPU selector replay contracts; no tensor, model or GPU access.

Scores may be dense offline oracle inputs. Online consumers receive only causal
absolute KV positions. Digests detect changes, not a producer's authenticity;
without score rows validation cannot certify a selection's score ranking.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol

from .schema import ValidationError

SELECTOR_MANIFEST_VERSION = "janus.selector.replay.v1"
MAX_SCORE_ELEMENTS = 10_000_000
MAX_CONFIG_FILE_BYTES = 1024 * 1024
_MANIFEST_FIELDS = {"schema_version", "config", "scores_sha256", "sequence_sha256", "steps", "manifest_sha256"}
_STEP_FIELDS = {"step_index", "step_id", "query_position", "cache_length", "target_token_id", "selections"}
_SELECTION_FIELDS = {"layer_index", "query_head_index", "kv_head_index", "absolute_kv_positions"}
_SCORE_ROW_FIELDS = {"step_index", "layer_index", "query_head_index", "scores"}


def _exact(value: Any, names: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value) or set(value) != names:
        raise ValidationError(f"{label} fields must equal {sorted(names)}")
    return value


def _int(value: Any, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValidationError(f"{label} must be a builtin integer in the declared bounds")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValidationError(f"{label} must be nonempty text")
    try:
        if len(value.encode("utf-8")) > 1024:
            raise ValidationError(f"{label} exceeds the text budget")
    except UnicodeError as exc:
        raise ValidationError(f"{label} must be valid UTF-8") from exc
    return value


def _hex(value: Any, length: int, label: str, *, lowercase: bool = False) -> str:
    alphabet = "0123456789abcdef" if lowercase else "0123456789abcdefABCDEF"
    if type(value) is not str or len(value) != length or any(char not in alphabet for char in value):
        raise ValidationError(f"{label} must be {length} hexadecimal characters")
    return value


def _finite_number(value: Any, label: str) -> None:
    if type(value) not in (int, float):
        raise ValidationError(f"{label} must be a finite builtin number")
    try:
        valid = math.isfinite(value)
    except (TypeError, OverflowError):
        valid = False
    if not valid:
        raise ValidationError(f"{label} must be a finite builtin number")


def _sha256(value: Any) -> str:
    digest = hashlib.sha256()
    try:
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        for part in encoder.iterencode(value):
            digest.update(part.encode("utf-8"))
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError("selector digest requires finite primitive UTF-8 JSON") from exc
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate selector config JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class SelectorReplayConfig:
    run_id: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    model_config_sha256: str
    weights_sha256: str | None
    source_kind: str
    seed: int
    prompt_token_ids: list[int] | tuple[int, ...]
    teacher_forced_token_ids: list[int] | tuple[int, ...]
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    top_k: int
    include_latest: bool = True
    selector_name: str = "score_topk"
    selector_version: str = "1"

    def __post_init__(self) -> None:
        for name, maximum in (("prompt_token_ids", 256), ("teacher_forced_token_ids", 32)):
            value = getattr(self, name)
            if type(value) in (list, tuple) and len(value) <= maximum:
                object.__setattr__(self, name, tuple(value))

    @property
    def cache_capacity(self) -> int:
        self.validate()
        return len(self.prompt_token_ids) + len(self.teacher_forced_token_ids) - 1

    def validate(self) -> None:
        _hex(self.run_id, 32, "run_id", lowercase=True)
        for name in ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision", "selector_name", "selector_version"):
            _text(getattr(self, name), name)
        _hex(self.model_config_sha256, 64, "model_config_sha256")
        if self.weights_sha256 is not None:
            _hex(self.weights_sha256, 64, "weights_sha256")
        if type(self.source_kind) is not str or self.source_kind not in ("synthetic_tensor_fixture", "offline_attention_scores"):
            raise ValidationError("unsupported selector source_kind")
        if self.source_kind == "offline_attention_scores" and self.weights_sha256 is None:
            raise ValidationError("offline_attention_scores requires weights_sha256 evidence")
        _int(self.seed, "seed")
        for name, maximum in (("prompt_token_ids", 256), ("teacher_forced_token_ids", 32)):
            tokens = getattr(self, name)
            if type(tokens) not in (list, tuple) or not 1 <= len(tokens) <= maximum:
                raise ValidationError(f"{name} must be a nonempty bounded list/tuple")
            for token in tokens:
                _int(token, name)
        for name, maximum in (("layers", 32), ("query_heads", 128), ("kv_heads", 128), ("head_dim", 256)):
            _int(getattr(self, name), name, 1, maximum)
        if self.query_heads % self.kv_heads:
            raise ValidationError("query_heads must be divisible by kv_heads")
        _int(self.top_k, "top_k", 1)
        if type(self.include_latest) is not bool:
            raise ValidationError("include_latest must be boolean")
        p, g = len(self.prompt_token_ids), len(self.teacher_forced_token_ids)
        if self.layers * self.query_heads * (g * p + g * (g - 1) // 2) > MAX_SCORE_ELEMENTS:
            raise ValidationError("selector score element budget exceeded")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["prompt_token_ids"] = list(self.prompt_token_ids)
        result["teacher_forced_token_ids"] = list(self.teacher_forced_token_ids)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SelectorReplayConfig":
        _exact(value, {field.name for field in fields(cls)}, "selector config")
        config = cls(**value)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "SelectorReplayConfig":
        source = Path(path)
        if source.stat().st_size > MAX_CONFIG_FILE_BYTES:
            raise ValidationError("selector config byte budget exceeded")
        with source.open("rb") as handle:
            raw = handle.read(MAX_CONFIG_FILE_BYTES + 1)
        if len(raw) > MAX_CONFIG_FILE_BYTES:
            raise ValidationError("selector config byte budget exceeded")
        def reject_constant(value: str) -> None:
            raise ValidationError(f"invalid nonfinite selector JSON number: {value}")
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=reject_constant)
        except (UnicodeError, ValueError, RecursionError) as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError(f"invalid selector config JSON: {exc}") from exc
        return cls.from_dict(value)


class TokenSelector(Protocol):
    name: str
    version: str

    def select(self, scores: Sequence[int | float], absolute_positions: Sequence[int],
               top_k: int, seed: int) -> Sequence[int]: ...


class DeterministicTopKSelector:
    name = "score_topk"
    version = "1"

    def select(self, scores: Sequence[int | float], absolute_positions: Sequence[int],
               top_k: int, seed: int) -> Sequence[int]:
        _int(top_k, "selector top_k", 1)
        _int(seed, "selector seed")
        if not isinstance(scores, Sequence) or not isinstance(absolute_positions, Sequence) or not scores or len(scores) != len(absolute_positions):
            raise ValidationError("selector scores/positions must be nonempty matching sequences")
        for score in scores:
            _finite_number(score, "selector score")
        for position in absolute_positions:
            _int(position, "selector absolute position")
        if len(set(absolute_positions)) != len(absolute_positions):
            raise ValidationError("selector absolute positions must be unique")
        ranked = sorted(zip(scores, absolute_positions), key=lambda pair: (-pair[0], pair[1]))
        # Score ranking decides membership; the replay contract uses position order.
        return tuple(sorted(position for _, position in ranked[:min(top_k, len(scores))]))


def _step_id(config: SelectorReplayConfig, index: int) -> str:
    return hashlib.sha256(f"{config.run_id}|step|{index}".encode("utf-8")).hexdigest()[:32]


def _sequence_sha256(config: SelectorReplayConfig) -> str:
    return _sha256({
        "prompt_token_ids": list(config.prompt_token_ids),
        "teacher_forced_token_ids": list(config.teacher_forced_token_ids),
        "alignment": {"step_index_base": 0, "absolute_position_base": 0,
                      "query_position_policy": "prompt_length_plus_step_index_minus_one",
                      "cache_length_policy": "prompt_length_plus_step_index",
                      "target_policy": "teacher_forced_token_ids_at_step_index",
                      "cache_includes_current_query": True},
    })


def _validated_score_rows(config: SelectorReplayConfig, score_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_count = len(config.teacher_forced_token_ids) * config.layers * config.query_heads
    try:
        iterator = iter(score_rows)
    except TypeError as exc:
        raise ValidationError("score_rows must be an iterable of strict row objects") from exc
    rows = []
    seen = set()
    for row in iterator:
        if len(rows) >= expected_count:
            raise ValidationError("too many selector score rows")
        _exact(row, _SCORE_ROW_FIELDS, "selector score row")
        step = _int(row["step_index"], "score step_index", 0, len(config.teacher_forced_token_ids) - 1)
        layer = _int(row["layer_index"], "score layer_index", 0, config.layers - 1)
        head = _int(row["query_head_index"], "score query_head_index", 0, config.query_heads - 1)
        key = (step, layer, head)
        if key in seen:
            raise ValidationError("duplicate selector score row")
        seen.add(key)
        scores = row["scores"]
        if type(scores) is not list or len(scores) != len(config.prompt_token_ids) + step:
            raise ValidationError("score row length must match causal cache length; future/absent scores are forbidden")
        for value in scores:
            _finite_number(value, "score row value")
        rows.append({"step_index": step, "layer_index": layer, "query_head_index": head, "scores": list(scores)})
    if len(rows) != expected_count:
        raise ValidationError("missing selector score rows")
    return sorted(rows, key=lambda row: (row["step_index"], row["layer_index"], row["query_head_index"]))


def _positions(value: Any, length: int, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValidationError(f"{label} must be a nonempty position sequence")
    if len(value) > length:
        raise ValidationError(f"{label} exceeds causal position count")
    positions = list(value)
    for position in positions:
        _int(position, label, 0, length - 1)
    if positions != sorted(set(positions)):
        raise ValidationError(f"{label} must be sorted and unique; reordered/duplicate positions are forbidden")
    return positions


def build_selector_manifest(config: SelectorReplayConfig, score_rows: Iterable[dict[str, Any]],
                            selector: TokenSelector | None = None) -> dict[str, Any]:
    if not isinstance(config, SelectorReplayConfig):
        raise ValidationError("build_selector_manifest requires SelectorReplayConfig")
    config.validate()
    strategy = DeterministicTopKSelector() if selector is None else selector
    strategy_name, strategy_version = getattr(strategy, "name", None), getattr(strategy, "version", None)
    if type(strategy_name) is not str or type(strategy_version) is not str or strategy_name != config.selector_name or strategy_version != config.selector_version:
        raise ValidationError("selector name/version must match config")
    select = getattr(strategy, "select", None)
    if not callable(select):
        raise ValidationError("selector must provide a callable select method")
    rows = _validated_score_rows(config, score_rows)
    scores_digest = _sha256(rows)
    steps = []
    for step_index, target in enumerate(config.teacher_forced_token_ids):
        length = len(config.prompt_token_ids) + step_index
        steps.append({"step_index": step_index, "step_id": _step_id(config, step_index),
                      "query_position": length - 1, "cache_length": length,
                      "target_token_id": target, "selections": []})
    for row in rows:
        step = steps[row["step_index"]]
        length = step["cache_length"]
        count = min(config.top_k, length)
        try:
            selected = select(tuple(row["scores"]), tuple(range(length)), count, config.seed)
        except Exception as exc:
            raise ValidationError(f"selector execution failed: {type(exc).__name__}") from exc
        positions = _positions(selected, length, "custom selector output")
        if len(positions) != count:
            raise ValidationError("selector output count must equal min(top_k, cache_length)")
        if config.include_latest and length - 1 not in positions:
            positions.append(length - 1)
        step["selections"].append({"layer_index": row["layer_index"], "query_head_index": row["query_head_index"],
                                   "kv_head_index": row["query_head_index"] // (config.query_heads // config.kv_heads),
                                   "absolute_kv_positions": positions})
    manifest = {"schema_version": SELECTOR_MANIFEST_VERSION, "config": config.to_dict(),
                "scores_sha256": scores_digest, "sequence_sha256": _sequence_sha256(config), "steps": steps}
    manifest["manifest_sha256"] = _sha256(manifest)
    return validate_selector_manifest(manifest, expected_config=config)


def validate_selector_manifest(manifest: dict[str, Any], expected_config: SelectorReplayConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate causal replay, provenance, layout, counts and self-consistency.

    The score rows are not embedded. This verifies neither their original score
    ranking nor malicious-producer authenticity. Supply expected_config to bind
    the manifest to the independently chosen model/sequence/layout contract.
    """
    _exact(manifest, _MANIFEST_FIELDS, "selector manifest")
    if type(manifest["schema_version"]) is not str or manifest["schema_version"] != SELECTOR_MANIFEST_VERSION:
        raise ValidationError("invalid selector manifest schema_version")
    config = SelectorReplayConfig.from_dict(manifest["config"])
    if expected_config is not None:
        expected = expected_config if isinstance(expected_config, SelectorReplayConfig) else SelectorReplayConfig.from_dict(expected_config)
        expected.validate()
        if _sha256(config.to_dict()) != _sha256(expected.to_dict()):
            raise ValidationError("selector manifest expected_config mismatch")
    _hex(manifest["scores_sha256"], 64, "scores_sha256")
    _hex(manifest["sequence_sha256"], 64, "sequence_sha256")
    if manifest["sequence_sha256"] != _sequence_sha256(config):
        raise ValidationError("selector sequence digest mismatch")
    steps = manifest["steps"]
    if type(steps) is not list or len(steps) != len(config.teacher_forced_token_ids):
        raise ValidationError("selector manifest must contain every teacher-forced step")
    for index, step in enumerate(steps):
        _exact(step, _STEP_FIELDS, "selector step")
        length = len(config.prompt_token_ids) + index
        if _int(step["step_index"], "step_index") != index:
            raise ValidationError("selector steps must be complete and in canonical step order")
        _hex(step["step_id"], 32, "step_id", lowercase=True)
        if step["step_id"] != _step_id(config, index):
            raise ValidationError("selector opaque step_id mismatch")
        if _int(step["query_position"], "query_position") != length - 1 or _int(step["cache_length"], "cache_length", 1) != length:
            raise ValidationError("selector step query/cache causal alignment mismatch")
        if _int(step["target_token_id"], "target_token_id") != config.teacher_forced_token_ids[index]:
            raise ValidationError("selector teacher-forced target mismatch")
        selections = step["selections"]
        if type(selections) is not list or len(selections) != config.layers * config.query_heads:
            raise ValidationError("selector step must contain every layer/query-head selection")
        for selection_index, selection in enumerate(selections):
            _exact(selection, _SELECTION_FIELDS, "selector selection")
            layer, head = divmod(selection_index, config.query_heads)
            if _int(selection["layer_index"], "selection layer_index") != layer or _int(selection["query_head_index"], "selection query_head_index") != head:
                raise ValidationError("selector selections must be complete and in canonical layer/head order")
            if _int(selection["kv_head_index"], "kv_head_index") != head // (config.query_heads // config.kv_heads):
                raise ValidationError("selector GQA KV-head mapping mismatch")
            if type(selection["absolute_kv_positions"]) is not list:
                raise ValidationError("manifest positions must be a builtin JSON list")
            positions = _positions(selection["absolute_kv_positions"], length, "manifest positions")
            count = min(config.top_k, length)
            if config.include_latest:
                if length - 1 not in positions or len(positions) not in (count, min(count + 1, length)):
                    raise ValidationError("selector include_latest/count policy mismatch")
            elif len(positions) != count:
                raise ValidationError("selector selection count must equal min(top_k, cache_length)")
    _hex(manifest["manifest_sha256"], 64, "manifest_sha256")
    if manifest["manifest_sha256"] != _sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"}):
        raise ValidationError("selector manifest digest mismatch")
    return deepcopy(manifest)
