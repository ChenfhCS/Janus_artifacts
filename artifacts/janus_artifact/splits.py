"""Deterministic group-isolated split and train-only vocabulary contracts."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Iterable

from .schema import ValidationError, validate_group_isolation


def _allocation(group_count: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: group_count * ratio for name, ratio in ratios.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remaining = group_count - sum(counts.values())
    order = sorted(ratios, key=lambda name: (-(raw[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def assign_grouped_splits(
    records: Iterable[dict[str, Any]],
    *,
    group_field: str = "case_id",
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: str = "janus-v1",
) -> list[dict[str, Any]]:
    ratios = {
        "train": train_ratio,
        "validation": validation_ratio,
        "test": test_ratio,
    }
    if any(value < 0 for value in ratios.values()):
        raise ValidationError("split ratios must be non-negative")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValidationError("split ratios must sum to 1")
    materialized = [deepcopy(record) for record in records]
    groups: set[str] = set()
    for record in materialized:
        group = record.get(group_field)
        if not isinstance(group, str) or not group:
            raise ValidationError(f"{group_field} must be a non-empty string")
        groups.add(group)
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    counts = _allocation(len(ordered), ratios)
    group_to_split: dict[str, str] = {}
    offset = 0
    for split in ("train", "validation", "test"):
        for group in ordered[offset : offset + counts[split]]:
            group_to_split[group] = split
        offset += counts[split]
    for record in materialized:
        record["split"] = group_to_split[record[group_field]]
    validate_group_isolation(materialized, group_field=group_field)
    return materialized


def freeze_vocabulary(
    records: Iterable[dict[str, Any]], *, token_path: tuple[str, str] = ("labels", "tokens")
) -> dict[str, Any]:
    tokens: set[str] = set()
    training_records = 0
    for record in records:
        if record.get("split") != "train":
            continue
        training_records += 1
        container = record.get(token_path[0], {})
        values = container.get(token_path[1], []) if isinstance(container, dict) else []
        if not isinstance(values, list) or not all(isinstance(token, str) for token in values):
            raise ValidationError("training labels.tokens must be an array of strings")
        tokens.update(values)
    if training_records == 0:
        raise ValidationError("cannot freeze vocabulary without training records")
    ordered = sorted(tokens)
    canonical = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)
    return {
        "schema_version": "janus.vocabulary.v1",
        "source_split": "train",
        "frozen": True,
        "tokens": ordered,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def validate_frozen_vocabulary(vocabulary: dict[str, Any]) -> set[str]:
    if vocabulary.get("schema_version") != "janus.vocabulary.v1":
        raise ValidationError("invalid vocabulary schema_version")
    if vocabulary.get("source_split") != "train" or vocabulary.get("frozen") is not True:
        raise ValidationError("vocabulary must be frozen from train split")
    tokens = vocabulary.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValidationError("vocabulary tokens must be an array of strings")
    if tokens != sorted(set(tokens)):
        raise ValidationError("vocabulary tokens must be unique and sorted")
    canonical = json.dumps(tokens, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if vocabulary.get("sha256") != digest:
        raise ValidationError("vocabulary digest mismatch")
    return set(tokens)
