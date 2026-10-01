"""Runnable QAI profiling pipeline with explicit reconstruction choices."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .legacy_npz import DEFAULT_MAX_FILE_BYTES, inspect_prefill_rank_npz
from .metrics import compute_pasr
from .schema import ValidationError, load_jsonl, write_jsonl


QAI_CHECKPOINT_VERSION = "janus.qai.state-dict.v2"
QAI_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class QAIConfig:
    """All values absent from the paper are explicit reconstruction choices."""

    architecture: str = "resnet18"
    selected_ranks: int = 10
    rank_selection: str = "largest_values"
    token_width: int = 0
    token_length_policy: str = "pad_to_train_max_reject_longer"
    base_channels: int = 64
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 1337
    num_workers: int = 0

    def validate(self) -> None:
        if self.architecture != "resnet18":
            raise ValidationError("this milestone implements resnet18 only")
        for name in ("selected_ranks", "base_channels", "batch_size", "epochs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValidationError(f"QAI config {name} must be a positive integer")
        if self.batch_size < 2:
            raise ValidationError("QAI config batch_size must be at least two for BatchNorm training")
        if self.rank_selection != "largest_values":
            raise ValidationError(
                "rank_selection must be largest_values, matching the statically inspected artifact"
            )
        if not isinstance(self.token_width, int) or isinstance(self.token_width, bool):
            raise ValidationError("QAI config token_width must be an integer")
        if self.token_width < 0:
            raise ValidationError("QAI config token_width must be zero or positive")
        if self.token_length_policy != "pad_to_train_max_reject_longer":
            raise ValidationError(
                "token_length_policy must be pad_to_train_max_reject_longer"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValidationError("QAI config seed must be an integer")
        if not isinstance(self.num_workers, int) or isinstance(self.num_workers, bool) or self.num_workers != 0:
            raise ValidationError("QAI num_workers must remain zero for deterministic loading")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValidationError(f"QAI config {name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValidationError(f"QAI config {name} must be finite and nonnegative")
        if self.learning_rate <= 0:
            raise ValidationError("QAI learning_rate must be positive")

    @classmethod
    def from_json(cls, path: str | Path) -> "QAIConfig":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid QAI config JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValidationError("QAI config must be a JSON object")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValidationError(f"unknown QAI config fields: {sorted(unknown)}")
        config = cls(**value)
        config.validate()
        return config


@dataclass(frozen=True)
class QAIRecord:
    sample_id: str
    case_id: str
    split: str
    attribute: str
    attribute_present: bool
    label: str | None
    npz_path: str


def _require_nonempty(record: dict[str, Any], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"QAI record {line_number} field {field} must be non-empty")
    return value


def load_qai_records(path: str | Path) -> list[QAIRecord]:
    """Load strict ID-keyed QAI records; row order has no semantic role."""
    source = Path(path)
    raw_records = load_jsonl(source)
    if not raw_records:
        raise ValidationError("QAI manifest must contain at least one record")
    records: list[QAIRecord] = []
    sample_ids: set[str] = set()
    case_splits: dict[str, set[str]] = {}
    for line_number, raw in enumerate(raw_records, start=1):
        expected = {
            "sample_id",
            "case_id",
            "split",
            "attribute",
            "attribute_present",
            "label",
            "npz_path",
        }
        if set(raw) != expected:
            raise ValidationError(
                f"QAI record {line_number} fields must equal {sorted(expected)}"
            )
        sample_id = _require_nonempty(raw, "sample_id", line_number)
        case_id = _require_nonempty(raw, "case_id", line_number)
        split = _require_nonempty(raw, "split", line_number)
        if split not in QAI_SPLITS:
            raise ValidationError(f"QAI record {line_number} has unsupported split {split}")
        if sample_id in sample_ids:
            raise ValidationError(f"duplicate QAI sample_id: {sample_id}")
        sample_ids.add(sample_id)
        case_splits.setdefault(case_id, set()).add(split)
        attribute_present = raw.get("attribute_present")
        if not isinstance(attribute_present, bool):
            raise ValidationError(
                f"QAI record {line_number} attribute_present must be boolean"
            )
        label = raw.get("label")
        if attribute_present:
            if not isinstance(label, str) or not label:
                raise ValidationError(
                    f"QAI record {line_number} requires a label when the attribute is present"
                )
        elif label is not None:
            raise ValidationError(
                f"QAI record {line_number} label must be null when the attribute is absent"
            )
        payload = Path(_require_nonempty(raw, "npz_path", line_number))
        if not payload.is_absolute():
            payload = (source.parent / payload).resolve()
        records.append(
            QAIRecord(
                sample_id=sample_id,
                case_id=case_id,
                split=split,
                attribute=_require_nonempty(raw, "attribute", line_number),
                attribute_present=attribute_present,
                label=label,
                npz_path=str(payload),
            )
        )
    leaked = {case: splits for case, splits in case_splits.items() if len(splits) > 1}
    if leaked:
        raise ValidationError(f"QAI case split leakage: {sorted(leaked)}")
    _qai_record_snapshot(records)
    return records


def freeze_qai_labels(records: Iterable[QAIRecord]) -> tuple[list[str], dict[str, int]]:
    records = list(records)
    attributes = {record.attribute for record in records}
    if len(attributes) != 1:
        raise ValidationError("one QAI run must target exactly one attribute")
    labels = sorted(
        {
            record.label
            for record in records
            if record.split == "train" and record.attribute_present
        }
    )
    if len(labels) < 2:
        raise ValidationError("QAI training requires at least two train labels")
    unseen = sorted(
        {
            record.label
            for record in records
            if record.attribute_present and record.label not in labels
        }
    )
    if unseen:
        raise ValidationError(f"non-train QAI labels are outside the frozen vocabulary: {unseen}")
    return labels, {label: index for index, label in enumerate(labels)}


def prefill_rank_feature(
    payload_path: str | Path,
    *,
    selected_ranks: int,
    rank_selection: str = "largest_values",
    target_token_width: int | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Convert a safe legacy rank tensor to (token, layer, head) features."""
    if not isinstance(selected_ranks, int) or isinstance(selected_ranks, bool) or selected_ranks <= 0:
        raise ValidationError("selected_ranks must be a positive integer")
    if rank_selection != "largest_values":
        raise ValidationError(
            "rank_selection must be largest_values, matching the statically inspected artifact"
        )
    if target_token_width is not None and (
        not isinstance(target_token_width, int)
        or isinstance(target_token_width, bool)
        or target_token_width <= 0
    ):
        raise ValidationError("target_token_width must be a positive integer when supplied")
    inspection = inspect_prefill_rank_npz(payload_path)
    try:
        with np.load(payload_path, allow_pickle=False) as payload:
            attn_rank = payload["attn_rank"]
            layer_count, head_count, query_count, token_count = attn_rank.shape
            rank_count = min(selected_ranks, token_count)
            selected = np.argpartition(attn_rank, -rank_count, axis=-1)[..., -rank_count:]
            counts = np.zeros((layer_count, head_count, token_count), dtype=np.float32)
            for layer in range(layer_count):
                for head in range(head_count):
                    counts[layer, head] = np.bincount(
                        selected[layer, head].reshape(-1), minlength=token_count
                    ).astype(np.float32)
    except ValueError as exc:
        raise ValidationError("QAI NPZ contains an unsupported array") from exc

    minimum = counts.min(axis=-1, keepdims=True)
    maximum = counts.max(axis=-1, keepdims=True)
    span = maximum - minimum
    normalized = np.divide(
        counts - minimum,
        span,
        out=np.zeros_like(counts),
        where=span > 0,
    )
    model_token_width = token_count if target_token_width is None else target_token_width
    if token_count > model_token_width:
        raise ValidationError(
            f"QAI token width {token_count} exceeds train-derived width {model_token_width}"
        )
    if token_count < model_token_width:
        normalized = np.pad(
            normalized,
            ((0, 0), (0, 0), (0, model_token_width - token_count)),
            mode="constant",
            constant_values=0,
        )
    feature_array = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    feature = torch.from_numpy(feature_array)
    metadata = {
        "source_sha256": inspection["sha256"],
        "source_shape": [layer_count, head_count, query_count, token_count],
        "feature_shape": list(feature.shape),
        "axis_contract": {
            "input": ["layer", "head", "query_token", "key_token"],
            "histogram_reduction_axes": ["query_token", "selected_rank_slot"],
            "preserved_axes": ["key_token", "layer", "head"],
            "output": ["key_token_channel", "layer_height", "head_width"],
        },
        "selected_ranks": rank_count,
        "rank_selection": rank_selection,
        "normalization": "per_layer_head_minmax",
        "normalization_fit_scope": "per_sample_no_dataset_statistics",
        "layout": "token_channel_layer_height_head_width",
        "natural_token_width": token_count,
        "model_token_width": model_token_width,
        "padded_token_positions": model_token_width - token_count,
        "feature_sha256": hashlib.sha256(feature_array.tobytes()).hexdigest(),
        "paper_claim": "scale normalization and cross-layer/head composition",
        "reconstruction_choice": (
            "artifact-informed largest-value rank histogram, per-layer/head minmax, "
            "KxLxH layout; rank polarity is not specified by the paper"
        ),
    }
    return feature, metadata


class QAIDataset(Dataset[tuple[Tensor, Tensor, str, str, str, str]]):
    def __init__(
        self,
        records: list[QAIRecord],
        label_to_index: dict[str, int],
        *,
        selected_ranks: int,
        rank_selection: str,
        expected_shape: tuple[int, int, int],
    ) -> None:
        if any(not record.attribute_present or record.label is None for record in records):
            raise ValidationError("QAI classifier dataset accepts present attributes only")
        self.records = records
        self.label_to_index = label_to_index
        self.selected_ranks = selected_ranks
        self.rank_selection = rank_selection
        self.expected_shape = expected_shape

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str, str, str, str]:
        record = self.records[index]
        feature, _ = prefill_rank_feature(
            record.npz_path,
            selected_ranks=self.selected_ranks,
            rank_selection=self.rank_selection,
            target_token_width=self.expected_shape[0],
        )
        if tuple(feature.shape) != self.expected_shape:
            raise ValidationError(
                f"QAI feature shape mismatch for {record.sample_id}: "
                f"{tuple(feature.shape)} != {self.expected_shape}"
            )
        assert record.label is not None
        target = torch.tensor(self.label_to_index[record.label], dtype=torch.long)
        return (
            feature,
            target,
            record.sample_id,
            record.case_id,
            record.attribute,
            record.label,
        )


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return self.relu(output + identity)


class QAIResNet18(nn.Module):
    """ResNet-18 depth with configurable width and token-position input channels."""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 64) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes < 2 or base_channels <= 0:
            raise ValidationError("invalid QAI ResNet dimensions")
        self.current_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_channels * 8, num_classes)
        self._initialize()

    def _make_layer(self, channels: int, *, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.current_channels, channels, stride)]
        self.current_channels = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.stem(inputs)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.pool(output)
        return self.classifier(torch.flatten(output, 1))


def _device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValidationError("QAI device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValidationError("CUDA was requested but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



QAI_PROVENANCE_VERSION = "janus.qai.training-provenance.v1"
QAI_PAYLOAD_SPLIT_POLICY = "reject_identical_payload_across_splits"


def _qai_record_snapshot(records: Iterable[QAIRecord]) -> list[dict[str, Any]]:
    """Validate identities first, then hash only classifier payloads."""
    records = list(records)
    if not records:
        raise ValidationError("QAI records must contain at least one record")
    sample_ids: set[str] = set()
    case_splits: dict[str, set[str]] = {}
    for record in records:
        if not isinstance(record, QAIRecord):
            raise ValidationError("QAI records must contain QAIRecord values")
        if any(not isinstance(value, str) or not value for value in (
            record.sample_id, record.case_id, record.attribute, record.npz_path
        )):
            raise ValidationError("QAI record identities and payload path must be non-empty strings")
        if record.split not in QAI_SPLITS:
            raise ValidationError("QAI record has an unsupported split")
        if record.sample_id in sample_ids:
            raise ValidationError(f"duplicate QAI sample_id: {record.sample_id}")
        sample_ids.add(record.sample_id)
        case_splits.setdefault(record.case_id, set()).add(record.split)
        if not isinstance(record.attribute_present, bool) or (
            record.attribute_present and (not isinstance(record.label, str) or not record.label)
        ) or (not record.attribute_present and record.label is not None):
            raise ValidationError("QAI record presence/label contract is invalid")
    if any(len(splits) > 1 for splits in case_splits.values()):
        raise ValidationError("QAI case split leakage")

    payload_splits: dict[str, set[str]] = {}
    rows = []
    for record in records:
        payload_digest = None
        if record.attribute_present:
            payload = Path(record.npz_path)
            try:
                if not payload.is_file() or not 0 < payload.stat().st_size <= DEFAULT_MAX_FILE_BYTES:
                    raise ValidationError(f"QAI payload is missing or exceeds the file-size limit: {record.sample_id}")
                payload_digest = _sha256(payload)
            except OSError as exc:
                raise ValidationError(f"QAI payload cannot be read: {record.sample_id}") from exc
            payload_splits.setdefault(payload_digest, set()).add(record.split)
        rows.append({
            "sample_id": record.sample_id,
            "case_id": record.case_id,
            "split": record.split,
            "attribute": record.attribute,
            "attribute_present": record.attribute_present,
            "label": record.label,
            "payload_sha256": payload_digest,
        })
    if any(len(splits) > 1 for splits in payload_splits.values()):
        raise ValidationError("QAI payload split leakage: identical payload bytes occur across splits")
    return sorted(rows, key=lambda row: row["sample_id"])


def _training_provenance(snapshot: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": QAI_PROVENANCE_VERSION,
        "payload_split_policy": QAI_PAYLOAD_SPLIT_POLICY,
        "absent_payload_policy": "not_loaded_when_attribute_absent",
        "records": [row for row in snapshot if row["split"] == "train"],
    }


def _validate_training_provenance(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    provenance = metadata.get("training_provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "version", "payload_split_policy", "absent_payload_policy", "records"
    } or (
        provenance.get("version") != QAI_PROVENANCE_VERSION
        or provenance.get("payload_split_policy") != QAI_PAYLOAD_SPLIT_POLICY
        or provenance.get("absent_payload_policy") != "not_loaded_when_attribute_absent"
    ):
        raise ValidationError("QAI checkpoint training provenance contract is invalid")
    rows = provenance["records"]
    if not isinstance(rows, list) or not rows:
        raise ValidationError("QAI checkpoint training provenance records are missing")
    ids: list[str] = []
    train_labels: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "sample_id", "case_id", "split", "attribute",
            "attribute_present", "label", "payload_sha256"
        }:
            raise ValidationError("QAI checkpoint training provenance record is invalid")
        if any(not isinstance(row[key], str) or not row[key] for key in (
            "sample_id", "case_id", "attribute"
        )) or row["split"] != "train" or row["attribute"] != metadata.get("attribute"):
            raise ValidationError("QAI checkpoint training provenance identity is invalid")
        ids.append(row["sample_id"])
        present = row["attribute_present"]
        label = row["label"]
        digest = row["payload_sha256"]
        if not isinstance(present, bool):
            raise ValidationError("QAI checkpoint training provenance presence is invalid")
        if present:
            if (
                not isinstance(label, str) or not label
                or not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValidationError("QAI checkpoint training provenance label/digest is invalid")
            train_labels.add(label)
        elif label is not None or digest is not None:
            raise ValidationError("QAI checkpoint absent training provenance must omit label/payload")
    if ids != sorted(set(ids)):
        raise ValidationError("QAI checkpoint training provenance IDs must be unique and sorted")
    if sorted(train_labels) != metadata.get("labels"):
        raise ValidationError("QAI checkpoint training provenance vocabulary mismatch")
    return rows


def _check_training_provenance(
    snapshot: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    training = _validate_training_provenance(metadata)
    by_sample = {row["sample_id"]: row for row in training}
    cases = {row["case_id"] for row in training}
    payloads = {row["payload_sha256"] for row in training if row["payload_sha256"] is not None}
    for row in snapshot:
        overlap = (
            row["sample_id"] in by_sample or row["case_id"] in cases
            or (row["payload_sha256"] is not None and row["payload_sha256"] in payloads)
        )
        if row["split"] != "train" and overlap:
            raise ValidationError(
                f"QAI training provenance overlap in non-train sample {row['sample_id']}"
            )
        if row["split"] == "train" and row["sample_id"] in by_sample and row != by_sample[row["sample_id"]]:
            raise ValidationError("QAI current train record differs from checkpoint training provenance")


def _training_batch_indices(
    sample_count: int, batch_size: int, generator: torch.Generator
) -> list[list[int]]:
    """Merge a final singleton into the preceding batch; retain every index."""
    if sample_count < 2 or batch_size < 2:
        raise ValidationError("QAI BatchNorm training requires two samples and batch_size at least two")
    indices = torch.randperm(sample_count, generator=generator).tolist()
    batches = [indices[start:start + batch_size] for start in range(0, sample_count, batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


class _QAITrainingBatchSampler:
    def __init__(self, sample_count: int, batch_size: int, generator: torch.Generator) -> None:
        self.sample_count = sample_count
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self):
        return iter(_training_batch_indices(self.sample_count, self.batch_size, self.generator))

    def __len__(self) -> int:
        count = math.ceil(self.sample_count / self.batch_size)
        return count - int(count > 1 and self.sample_count % self.batch_size == 1)


def _require_finite_tensor(value: Tensor, context: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValidationError(f"QAI {context} contains non-finite values")


def _validate_checkpoint_metadata(metadata: dict[str, Any]) -> QAIConfig:
    if metadata.get("checkpoint_version") != QAI_CHECKPOINT_VERSION:
        raise ValidationError("unsupported QAI checkpoint metadata")
    raw_config = metadata.get("config")
    if not isinstance(raw_config, dict) or set(raw_config) != set(QAIConfig.__dataclass_fields__):
        raise ValidationError("QAI checkpoint config fields are invalid")
    try:
        config = QAIConfig(**raw_config)
        config.validate()
    except (TypeError, ValueError) as exc:
        raise ValidationError("QAI checkpoint config is invalid") from exc
    shape = metadata.get("input_shape")
    labels = metadata.get("labels")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in shape)
        or not isinstance(labels, list)
        or len(labels) < 2
        or not all(isinstance(label, str) and label for label in labels)
    ):
        raise ValidationError("QAI checkpoint dimensions or labels are invalid")
    if labels != sorted(set(labels)):
        raise ValidationError("QAI checkpoint labels must be unique and sorted")
    if metadata.get("architecture") != config.architecture:
        raise ValidationError("QAI checkpoint architecture/config contract mismatch")
    if not isinstance(metadata.get("attribute"), str) or not metadata["attribute"]:
        raise ValidationError("QAI checkpoint attribute is invalid")
    natural_width = metadata.get("train_max_natural_token_width")
    if not isinstance(natural_width, int) or isinstance(natural_width, bool) or natural_width <= 0:
        raise ValidationError("QAI checkpoint train token width is invalid")
    expected_width = config.token_width or natural_width
    expected_source = "explicit_config" if config.token_width else "maximum_over_train_split_only"
    if (
        shape[0] != expected_width
        or natural_width > shape[0]
        or metadata.get("token_width_source") != expected_source
    ):
        raise ValidationError("QAI checkpoint token width/config contract mismatch")
    if metadata.get("standard_resnet_width") != (config.base_channels == 64):
        raise ValidationError("QAI checkpoint model width/config contract mismatch")
    feature_metadata = metadata.get("first_feature_metadata")
    if not isinstance(feature_metadata, dict) or (
        feature_metadata.get("feature_shape") != shape
        or feature_metadata.get("model_token_width") != shape[0]
        or feature_metadata.get("rank_selection") != config.rank_selection
    ):
        raise ValidationError("QAI checkpoint feature/config contract mismatch")
    _validate_training_provenance(metadata)
    return config


def _checkpoint_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    # Every metadata field, including provenance added by callers, is bound to the state.
    # These two digests are excluded to avoid a circular checksum dependency.
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"weights_sha256", "contract_sha256"}
    }


def _contract_sha256(contract: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError("QAI checkpoint contract must contain finite JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_qai_state_dict(state: Any, expected: dict[str, Tensor] | None = None) -> None:
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()
    ):
        raise ValidationError("QAI checkpoint must contain only a tensor state_dict")
    if expected is not None and set(state) != set(expected):
        raise ValidationError("QAI checkpoint state/config contract mismatch")
    for key, value in state.items():
        if value.layout != torch.strided:
            raise ValidationError("QAI checkpoint state tensors must have strided layout")
        _require_finite_tensor(value, f"checkpoint state tensor {key}")
        if expected is not None and (
            value.shape != expected[key].shape or value.dtype != expected[key].dtype
        ):
            raise ValidationError(f"QAI checkpoint state/config contract mismatch at {key}")


def _save_qai_checkpoint(
    weights_path: Path, metadata: dict[str, Any], state: dict[str, Tensor]
) -> None:
    """Bind JSON semantics and tensor state for integrity, not adversarial authentication."""
    _validate_checkpoint_metadata(metadata)
    _validate_qai_state_dict(state)
    if metadata.get("weights_file") != weights_path.name:
        raise ValidationError("QAI checkpoint weights filename mismatch")
    contract = _checkpoint_contract(metadata)
    contract_digest = _contract_sha256(contract)
    torch.save(
        {"contract": contract, "contract_sha256": contract_digest, "state_dict": state},
        weights_path,
    )
    metadata["contract_sha256"] = contract_digest
    metadata["weights_sha256"] = _sha256(weights_path)


def _evaluate_loss(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, int, int]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for features, targets, *_ in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            _require_finite_tensor(logits, "validation logits")
            loss = criterion(logits, targets)
            _require_finite_tensor(loss, "validation loss")
            total_loss += float(loss.item()) * targets.shape[0]
            total += targets.shape[0]
            correct += int((logits.argmax(dim=1) == targets).sum().item())
    return total_loss / total if total else 0.0, correct, total


def train_qai(
    records: list[QAIRecord],
    config: QAIConfig,
    checkpoint_dir: str | Path,
    *,
    device: str = "auto",
    run_kind: str = "unvalidated_research_run",
) -> dict[str, Any]:
    config.validate()
    record_snapshot = _qai_record_snapshot(records)
    labels, label_to_index = freeze_qai_labels(records)
    train_records = [
        record
        for record in records
        if record.split == "train" and record.attribute_present
    ]
    validation_records = [
        record
        for record in records
        if record.split == "validation" and record.attribute_present
    ]
    if not train_records:
        raise ValidationError("QAI training requires present-attribute train records")
    natural_shapes: list[tuple[int, int, int]] = []
    for record in train_records:
        feature, _ = prefill_rank_feature(
            record.npz_path,
            selected_ranks=config.selected_ranks,
            rank_selection=config.rank_selection,
        )
        natural_shapes.append(tuple(int(value) for value in feature.shape))
    spatial_shapes = {(shape[1], shape[2]) for shape in natural_shapes}
    if len(spatial_shapes) != 1:
        raise ValidationError(
            "QAI train records must share layer/head dimensions; no layer/head padding is inferred"
        )
    train_max_token_width = max(shape[0] for shape in natural_shapes)
    model_token_width = config.token_width or train_max_token_width
    if model_token_width < train_max_token_width:
        raise ValidationError(
            "configured QAI token_width is smaller than a training feature; truncation is forbidden"
        )
    layer_count, head_count = next(iter(spatial_shapes))
    expected_shape = (model_token_width, layer_count, head_count)
    _, first_metadata = prefill_rank_feature(
        train_records[0].npz_path,
        selected_ranks=config.selected_ranks,
        rank_selection=config.rank_selection,
        target_token_width=model_token_width,
    )
    _seed_everything(config.seed)
    resolved_device = _device(device)
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    train_dataset = QAIDataset(
        train_records,
        label_to_index,
        selected_ranks=config.selected_ranks,
        rank_selection=config.rank_selection,
        expected_shape=expected_shape,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=_QAITrainingBatchSampler(len(train_dataset), config.batch_size, generator),
        num_workers=config.num_workers,
    )
    validation_loader = None
    if validation_records:
        validation_loader = DataLoader(
            QAIDataset(
                validation_records,
                label_to_index,
                selected_ranks=config.selected_ranks,
                rank_selection=config.rank_selection,
                expected_shape=expected_shape,
            ),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
    model = QAIResNet18(
        in_channels=expected_shape[0],
        num_classes=len(labels),
        base_channels=config.base_channels,
    ).to(resolved_device)
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    first_parameter = next(model.parameters()).detach().cpu().clone()
    history: list[dict[str, Any]] = []
    gradient_observed = False
    started = time.monotonic()
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for features, targets, *_ in train_loader:
            features = features.to(resolved_device)
            targets = targets.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            _require_finite_tensor(logits, "training logits")
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValidationError("QAI training produced a non-finite loss")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    _require_finite_tensor(parameter.grad, "training gradient")
            gradient_observed = gradient_observed or any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all().item())
                and bool(torch.count_nonzero(parameter.grad).item())
                for parameter in model.parameters()
            )
            optimizer.step()
            total_loss += float(loss.item()) * targets.shape[0]
            total += targets.shape[0]
        epoch_result: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": total_loss / total,
            "train_total": total,
        }
        if validation_loader is not None:
            val_loss, val_correct, val_total = _evaluate_loss(
                model, validation_loader, criterion, resolved_device
            )
            epoch_result.update(
                {
                    "validation_loss": val_loss,
                    "validation_correct": val_correct,
                    "validation_total": val_total,
                }
            )
        history.append(epoch_result)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    duration_seconds = time.monotonic() - started
    parameter_changed = not torch.equal(first_parameter, next(model.parameters()).detach().cpu())
    if not gradient_observed or not parameter_changed:
        raise ValidationError("QAI smoke training did not observe a finite gradient update")

    if _qai_record_snapshot(records) != record_snapshot:
        raise ValidationError("QAI payloads changed during training; no checkpoint is saved")

    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "model_state.pt"
    metadata_path = destination / "metadata.json"
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    metadata = {
        "checkpoint_version": QAI_CHECKPOINT_VERSION,
        "weights_file": weights_path.name,
        "architecture": "resnet18",
        "config": asdict(config),
        "input_shape": list(expected_shape),
        "token_width_source": (
            "explicit_config" if config.token_width else "maximum_over_train_split_only"
        ),
        "train_max_natural_token_width": train_max_token_width,
        "labels": labels,
        "attribute": train_records[0].attribute,
        "training_provenance": _training_provenance(record_snapshot),
        "run_kind": run_kind,
        "paper_backed": [
            "supervised query-attribute classification",
            "scale normalization",
            "cross-layer and cross-head feature composition",
            "18-layer residual predictor",
        ],
        "reconstruction_choices": [
            "rank histogram feature",
            "per-layer/head minmax normalization",
            "token-channel x layer x head tensor layout",
            "short samples zero-padded to a train-only width; longer non-train samples rejected",
            "optimizer and all numerical hyperparameters",
        ],
        "first_feature_metadata": first_metadata,
        "model_parameter_count": model_parameter_count,
        "standard_resnet_width": config.base_channels == 64,
    }
    _save_qai_checkpoint(weights_path, metadata, cpu_state)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(resolved_device))
        if resolved_device.type == "cuda"
        else 0
    )
    return {
        "checkpoint_dir": str(destination),
        "device": str(resolved_device),
        "duration_seconds": duration_seconds,
        "peak_cuda_memory_bytes": peak_bytes,
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "absent_attribute_train_records_excluded": sum(
            record.split == "train" and not record.attribute_present for record in records
        ),
        "model_parameter_count": model_parameter_count,
        "standard_resnet_width": config.base_channels == 64,
        "model_input_shape": list(expected_shape),
        "token_width_source": metadata["token_width_source"],
        "gradient_observed": gradient_observed,
        "parameter_changed": parameter_changed,
        "history": history,
        "weights_sha256": metadata["weights_sha256"],
        "scientific_result": False,
    }


def load_qai_checkpoint(
    checkpoint_dir: str | Path, *, device: str = "auto"
) -> tuple[QAIResNet18, dict[str, Any], torch.device]:
    directory = Path(checkpoint_dir)
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid QAI checkpoint metadata: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("checkpoint_version") != QAI_CHECKPOINT_VERSION:
        raise ValidationError("unsupported QAI checkpoint metadata")
    config = _validate_checkpoint_metadata(metadata)
    contract = _checkpoint_contract(metadata)
    contract_digest = _contract_sha256(contract)
    if metadata.get("contract_sha256") != contract_digest:
        raise ValidationError("QAI checkpoint metadata contract digest mismatch")
    weights_file = metadata.get("weights_file")
    if (
        not isinstance(weights_file, str)
        or not weights_file
        or Path(weights_file).name != weights_file
        or weights_file in {".", ".."}
    ):
        raise ValidationError("QAI checkpoint weights filename must be local")
    weights_path = directory / weights_file
    if not weights_path.is_file() or _sha256(weights_path) != metadata.get("weights_sha256"):
        raise ValidationError("QAI checkpoint weight digest mismatch")
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValidationError("QAI safe state_dict loading failed") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "contract", "contract_sha256", "state_dict"
    }:
        raise ValidationError("QAI checkpoint requires a bound metadata/state contract")
    saved_contract = checkpoint["contract"]
    if (
        not isinstance(saved_contract, dict)
        or _contract_sha256(saved_contract) != contract_digest
        or checkpoint["contract_sha256"] != contract_digest
    ):
        raise ValidationError("QAI checkpoint metadata/state contract mismatch")
    resolved_device = _device(device)
    model = QAIResNet18(metadata["input_shape"][0], len(metadata["labels"]), config.base_channels)
    if metadata.get("model_parameter_count") != sum(parameter.numel() for parameter in model.parameters()):
        raise ValidationError("QAI checkpoint parameter count/config contract mismatch")
    state = checkpoint["state_dict"]
    _validate_qai_state_dict(state, model.state_dict())
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValidationError("QAI checkpoint state/config contract mismatch") from exc
    model.to(resolved_device).eval()
    return model, metadata, resolved_device


def predict_qai(
    records: list[QAIRecord],
    checkpoint_dir: str | Path,
    *,
    split: str,
    device: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split not in QAI_SPLITS:
        raise ValidationError(f"unsupported QAI inference split: {split}")
    selected_records = [record for record in records if record.split == split]
    if not selected_records:
        raise ValidationError(f"QAI inference split {split} has no records")
    present_records = [record for record in selected_records if record.attribute_present]
    model, metadata, resolved_device = load_qai_checkpoint(checkpoint_dir, device=device)
    if any(record.attribute != metadata["attribute"] for record in records):
        raise ValidationError("QAI inference attribute does not match checkpoint attribute")
    record_snapshot = _qai_record_snapshot(records)
    _check_training_provenance(record_snapshot, metadata)
    labels = metadata["labels"]
    label_to_index = {label: index for index, label in enumerate(labels)}
    unseen = sorted(
        {
            record.label
            for record in present_records
            if record.label not in label_to_index
        }
    )
    if unseen:
        raise ValidationError(f"QAI inference labels are outside checkpoint vocabulary: {unseen}")
    config = QAIConfig(**metadata["config"])
    predictions_by_id: dict[str, dict[str, Any]] = {
        record.sample_id: {
            "record_id": record.sample_id,
            "case_id": record.case_id,
            "split": split,
            "attribute": record.attribute,
            "attribute_present": False,
            "gold_value": None,
            "predicted_value": None,
            "predicted_probability": None,
            "classification_skipped": "attribute_absent",
        }
        for record in selected_records
        if not record.attribute_present
    }
    model.eval()
    if present_records:
        loader = DataLoader(
            QAIDataset(
                present_records,
                label_to_index,
                selected_ranks=config.selected_ranks,
                rank_selection=config.rank_selection,
                expected_shape=tuple(metadata["input_shape"]),
            ),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        with torch.no_grad():
            for features, _, sample_ids, case_ids, attributes, gold_labels in loader:
                logits = model(features.to(resolved_device))
                _require_finite_tensor(logits, "inference logits")
                probabilities = torch.softmax(logits, dim=1).cpu()
                _require_finite_tensor(probabilities, "inference probabilities")
                predicted = probabilities.argmax(dim=1)
                for offset, predicted_index in enumerate(predicted.tolist()):
                    predictions_by_id[sample_ids[offset]] = {
                        "record_id": sample_ids[offset],
                        "case_id": case_ids[offset],
                        "split": split,
                        "attribute": attributes[offset],
                        "attribute_present": True,
                        "gold_value": gold_labels[offset],
                        "predicted_value": labels[predicted_index],
                        "predicted_probability": float(
                            probabilities[offset, predicted_index].item()
                        ),
                        "classification_skipped": None,
                    }
    predictions = [predictions_by_id[record.sample_id] for record in selected_records]
    metrics = compute_pasr(predictions)
    return predictions, metrics


def write_qai_predictions(path: str | Path, predictions: Iterable[dict[str, Any]]) -> None:
    write_jsonl(path, predictions)


def create_synthetic_qai_fixture(root: str | Path, *, seed: int = 7) -> Path:
    """Create a tiny learnability fixture; never use its PASR as a scientific result."""
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    split_counts = {"train": 12, "validation": 4, "test": 4}
    sample_number = 0
    for split, count in split_counts.items():
        for within_split in range(count):
            label_index = within_split % 2
            label = f"synthetic-class-{label_index}"
            token_count = 12 + 2 * (within_split % 3)
            attn_rank = rng.integers(
                0, 32, size=(8, 8, 3, token_count), dtype=np.uint16
            )
            key = 2 if label_index == 0 else token_count - 3
            attn_rank[..., key] = np.uint16(256)
            attn_rank[..., (key + 1) % token_count] = np.uint16(255)
            payload = destination / f"synthetic-{sample_number}.npz"
            np.savez_compressed(
                payload,
                attn_rank=attn_rank,
                top_k=np.asarray(256, dtype=np.uint16),
            )
            records.append(
                {
                    "sample_id": f"synthetic-sample-{sample_number}",
                    "case_id": f"synthetic-case-{sample_number}",
                    "split": split,
                    "attribute": "synthetic-attribute",
                    "attribute_present": True,
                    "label": label,
                    "npz_path": payload.name,
                }
            )
            sample_number += 1
    records.extend(
        [
            {
                "sample_id": "synthetic-absent-train",
                "case_id": "synthetic-absent-train",
                "split": "train",
                "attribute": "synthetic-attribute",
                "attribute_present": False,
                "label": None,
                "npz_path": "absent-train-payload-is-not-read.npz",
            },
            {
                "sample_id": "synthetic-absent-test",
                "case_id": "synthetic-absent-test",
                "split": "test",
                "attribute": "synthetic-attribute",
                "attribute_present": False,
                "label": None,
                "npz_path": "absent-test-payload-is-not-read.npz",
            },
        ]
    )
    manifest = destination / "manifest.jsonl"
    write_jsonl(manifest, records)
    return manifest


def run_synthetic_qai_smoke(root: str | Path, *, device: str = "auto") -> dict[str, Any]:
    destination = Path(root)
    manifest_path = create_synthetic_qai_fixture(destination / "data")
    records = load_qai_records(manifest_path)
    config = QAIConfig(
        selected_ranks=2,
        base_channels=8,
        batch_size=4,
        epochs=2,
        learning_rate=3e-3,
        weight_decay=1e-4,
        seed=19,
    )
    training = train_qai(
        records,
        config,
        destination / "checkpoint",
        device=device,
        run_kind="synthetic_smoke_only",
    )
    predictions, pasr = predict_qai(
        records,
        destination / "checkpoint",
        split="test",
        device=device,
    )
    write_qai_predictions(destination / "predictions.jsonl", predictions)
    report = {
        "status": "ok",
        "synthetic_only": True,
        "scientific_result": False,
        "training": training,
        "test_predictions": len(predictions),
        "pasr_smoke_only": pasr,
        "checkpoint_reload_verified": True,
    }
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report
