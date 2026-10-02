"""Explicit reconstruction choices for causal ATR features and token batches."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .atr_data import validate_atr_records
from .schema import ValidationError


@dataclass(frozen=True)
class ATRConfig:
    architecture: str = "resnet18"
    token_width: int = 0
    base_channels: int = 64
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 1337
    num_workers: int = 0
    normalization: str = "none"
    sequential_augmentation: str = "causal_running_mean"
    augmentation_strength: float = 0.5

    def validate(self) -> None:
        if self.architecture != "resnet18":
            raise ValidationError("ATR implements the resnet18 reconstruction only")
        for name in ("base_channels", "batch_size", "epochs"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValidationError(f"ATR config {name} must be a positive integer")
        if self.batch_size < 2:
            raise ValidationError("ATR config batch_size must be at least two for BatchNorm")
        if type(self.token_width) is not int or self.token_width < 0:
            raise ValidationError("ATR token_width must be zero or a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValidationError("ATR seed must be an integer in [0, 2**32)")
        if type(self.num_workers) is not int or self.num_workers != 0:
            raise ValidationError("ATR num_workers must remain zero")
        for name in ("learning_rate", "weight_decay", "augmentation_strength"):
            value = getattr(self, name)
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except (OverflowError, TypeError):
                finite = False
            if not finite:
                raise ValidationError(f"ATR config {name} must be finite numeric")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValidationError("ATR learning_rate must be positive and weight_decay nonnegative")
        if not 0 <= self.augmentation_strength <= 1:
            raise ValidationError("ATR augmentation_strength must lie in [0, 1]")
        if self.normalization not in {"none", "per_layer_head_minmax"}:
            raise ValidationError("unsupported ATR normalization reconstruction")
        if self.sequential_augmentation not in {"none", "causal_running_mean"}:
            raise ValidationError("unsupported ATR sequential augmentation reconstruction")

    @classmethod
    def from_json(cls, path: str | Path) -> "ATRConfig":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError("ATR config must be readable JSON") from exc
        if not isinstance(raw, dict) or set(raw) - set(cls.__dataclass_fields__):
            raise ValidationError("ATR config has invalid or unknown fields")
        config = cls(**raw)
        config.validate()
        return config


def derive_atr_input_shape(
    train_records: list[dict[str, Any]], config: ATRConfig
) -> tuple[int, int, int]:
    """Choose token width from available train profiles only."""
    config.validate()
    records = validate_atr_records(train_records)
    if any(record["split"] != "train" for record in records):
        raise ValidationError("ATR input shape must be derived from train records only")
    widths = [
        len(step["profile"][0][0])
        for record in records
        for step in record["steps"]
        if step["profile"] is not None
    ]
    if not widths:
        raise ValidationError("ATR train profiles are missing")
    natural_width = max(widths)
    width = config.token_width or natural_width
    if width < natural_width:
        raise ValidationError("ATR configured token width would truncate train profiles")
    contract = records[0]["feature_contract"]
    return width, len(contract["layer_ids"]), len(contract["kv_head_ids"])


def prepare_atr_features(
    records: list[dict[str, Any]], config: ATRConfig, input_shape: tuple[int, int, int]
) -> dict[tuple[str, str], Tensor]:
    """Augment with the mean of earlier observed profiles in the same response.

    The running mean uses unaugmented profiles, excludes missing observations,
    and never uses token labels, previous predictions, or future profiles.
    """
    config.validate()
    records = validate_atr_records(records)
    if (
        not isinstance(input_shape, tuple)
        or len(input_shape) != 3
        or any(type(value) is not int or value <= 0 for value in input_shape)
    ):
        raise ValidationError("ATR input_shape must be three positive integer dimensions")
    width, layers, heads = input_shape
    features: dict[tuple[str, str], Tensor] = {}
    for record in records:
        contract = record["feature_contract"]
        if (len(contract["layer_ids"]), len(contract["kv_head_ids"])) != (layers, heads):
            raise ValidationError("ATR layer/head dimensions do not match the feature contract")
        history_sum = torch.zeros(input_shape, dtype=torch.float32)
        history_count = 0
        for step in record["steps"]:
            profile = step["profile"]
            if profile is None:
                continue
            natural_width = len(profile[0][0])
            if natural_width > width:
                raise ValidationError("ATR profile width exceeds train-derived width; truncation forbidden")
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.asarray(profile, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValidationError("ATR profile contains non-finite float32 values")
            if config.normalization == "per_layer_head_minmax":
                minimum = array.min(axis=-1, keepdims=True)
                span = array.max(axis=-1, keepdims=True) - minimum
                array = np.divide(array - minimum, span, out=np.zeros_like(array), where=span > 0)
            if natural_width < width:
                array = np.pad(array, ((0, 0), (0, 0), (0, width - natural_width)))
            current = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
            augmented = current
            if config.sequential_augmentation == "causal_running_mean" and history_count:
                augmented = current + config.augmentation_strength * (history_sum / history_count)
            if not bool(torch.isfinite(augmented).all().item()):
                raise ValidationError("ATR augmented features contain non-finite values")
            features[(record["response_id"], step["step_id"])] = augmented
            history_sum = history_sum + current
            history_count += 1
    return features


class ATRTokenDataset(Dataset):
    """Use only available profiles whose explicit gold token is in train vocabulary."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        features: dict[tuple[str, str], Tensor],
        token_to_index: dict[int, int],
    ) -> None:
        self.entries = [
            (record["response_id"], step)
            for record in (validate_atr_records(records) if records else [])
            for step in record["steps"]
            if (record["response_id"], step["step_id"]) in features
            and step["gold_token_id"] in token_to_index
        ]
        self.features = features
        self.token_to_index = token_to_index

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        response_id, step = self.entries[index]
        token = step["gold_token_id"]
        return (
            self.features[(response_id, step["step_id"])],
            torch.tensor(self.token_to_index[token], dtype=torch.long),
            response_id,
            step["step_id"],
            step["step_index"],
            token,
        )
