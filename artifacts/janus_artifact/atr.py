"""Step-aligned ATR training and inference with explicit reconstruction contracts.

The ResNet layout and causal mean augmentation are reconstruction choices, not
an exact implementation of the paper. Digests detect integrity mismatches; they
are not authentication against a party able to rewrite the checkpoint.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .atr_data import (
    atr_record_contract,
    atr_record_snapshot,
    freeze_atr_vocabulary,
    load_atr_records,
    validate_atr_records,
    validate_atr_vocabulary,
)
from .atr_features import (
    ATRConfig,
    ATRTokenDataset,
    derive_atr_input_shape,
    prepare_atr_features,
)
from .atr_metrics import compute_atr_dasr
from .qai import (
    QAIResNet18,
    _QAITrainingBatchSampler,
    _contract_sha256,
    _device,
    _seed_everything,
    _sha256,
    _validate_qai_state_dict,
)
from .schema import ValidationError, write_jsonl


ATR_CHECKPOINT_VERSION = "janus.atr.state-dict.v1"
ATR_PROVENANCE_VERSION = "janus.atr.training-provenance.v1"
ATR_SPLITS = {"train", "validation", "test"}
ATR_PROFILE_SPLIT_POLICY = "reject_float32_equivalent_complete_response_profiles_across_splits"
ATR_PROFILE_DIGEST_ENCODING = "float32_le_shape_missingmask_v1"


def _finite(value: Tensor, context: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValidationError(f"ATR {context} contains non-finite values")


def _same_json_values(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare canonical JSON while preserving boolean/integer/float types."""
    return _contract_sha256(left) == _contract_sha256(right)


def _run_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    contract = atr_record_contract(records[0])
    if any(not _same_json_values(atr_record_contract(record), contract) for record in records):
        raise ValidationError("ATR records must share task/tokenizer/alignment/feature contract")
    return contract


def _validate_record_contract(contract: Any) -> None:
    if not isinstance(contract, dict) or set(contract) != {
        "schema_version", "task_id", "tokenizer", "alignment", "feature_contract"
    }:
        raise ValidationError("ATR checkpoint record contract fields are invalid")
    # The ordinary data validator checks every semantic field without allocating
    # a profile tensor. A test record with no profile/gold is a legal pure-predict
    # input and cannot contribute to a metric or a training vocabulary.
    probe = {
        **contract,
        "response_id": "checkpoint-contract-probe",
        "case_id": "checkpoint-contract-probe",
        "split": "test",
        "source_kind": "synthetic_fixture",
        "provenance": {"synthetic": True},
        "steps": [{
            "step_id": "checkpoint-contract-probe-step",
            "step_index": 0,
            "gold_token_id": None,
            "profile": None,
        }],
    }
    try:
        checked = validate_atr_records([probe])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValidationError("ATR checkpoint record contract is invalid") from exc
    if not _same_json_values(atr_record_contract(checked[0]), contract):
        raise ValidationError("ATR checkpoint record contract is not canonical")


def _checkpoint_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in metadata.items()
        if key not in {"weights_sha256", "contract_sha256"}
    }


def _training_provenance(
    snapshot: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "version": ATR_PROVENANCE_VERSION,
        "profile_split_policy": ATR_PROFILE_SPLIT_POLICY,
        "profile_digest_encoding": ATR_PROFILE_DIGEST_ENCODING,
        "records": [row for row in snapshot if row["split"] == "train"],
        "sources": [{
            "response_id": record["response_id"],
            "source_kind": record["source_kind"],
            "provenance": record["provenance"],
        } for record in records if record["split"] == "train"],
    }


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_training_provenance(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    provenance = metadata.get("training_provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "version", "profile_split_policy", "profile_digest_encoding", "records", "sources"
    } or (
        provenance.get("version") != ATR_PROVENANCE_VERSION
        or provenance.get("profile_split_policy") != ATR_PROFILE_SPLIT_POLICY
        or provenance.get("profile_digest_encoding") != ATR_PROFILE_DIGEST_ENCODING
    ):
        raise ValidationError("ATR checkpoint training provenance contract is invalid")
    rows = provenance["records"]
    sources = provenance["sources"]
    if not isinstance(rows, list) or not rows or not isinstance(sources, list):
        raise ValidationError("ATR checkpoint training provenance records are missing")
    ids: list[str] = []
    step_ids: list[str] = []
    tokens: set[int] = set()
    candidates = validate_atr_vocabulary(metadata["candidate_vocabulary"], metadata["record_contract"])
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "response_id", "case_id", "split", "task_id", "step_ids",
            "gold_token_ids", "profile_sha256"
        }:
            raise ValidationError("ATR checkpoint training provenance record fields are invalid")
        if any(not isinstance(row[key], str) or not row[key] for key in (
            "response_id", "case_id", "task_id"
        )) or row["split"] != "train" or row["task_id"] != metadata["record_contract"]["task_id"]:
            raise ValidationError("ATR checkpoint training provenance identity is invalid")
        row_steps = row["step_ids"]
        gold = row["gold_token_ids"]
        if (
            not isinstance(row_steps, list) or not row_steps
            or not all(isinstance(value, str) and value for value in row_steps)
            or len(set(row_steps)) != len(row_steps)
            or not isinstance(gold, list) or len(gold) != len(row_steps)
            or not all(isinstance(value, int) and not isinstance(value, bool) and value in candidates for value in gold)
            or not _valid_digest(row["profile_sha256"])
        ):
            raise ValidationError("ATR checkpoint training provenance steps/tokens/digest are invalid")
        ids.append(row["response_id"])
        step_ids.extend(row_steps)
        tokens.update(gold)
    if ids != sorted(set(ids)) or len(step_ids) != len(set(step_ids)):
        raise ValidationError("ATR checkpoint training response/step IDs must be unique and responses sorted")
    if tokens != candidates:
        raise ValidationError("ATR checkpoint training vocabulary/provenance mismatch")
    if len(sources) != len(rows):
        raise ValidationError("ATR checkpoint training source coverage mismatch")
    for row, source in zip(rows, sources):
        if not isinstance(source, dict) or set(source) != {
            "response_id", "source_kind", "provenance"
        } or source["response_id"] != row["response_id"]:
            raise ValidationError("ATR checkpoint training source identity mismatch")
        # Check source/provenance using the same canonical validator as input.
        probe = {
            **metadata["record_contract"],
            "response_id": row["response_id"], "case_id": row["case_id"],
            "split": "test", "source_kind": source["source_kind"],
            "provenance": source["provenance"],
            "steps": [{
                "step_id": row["step_ids"][0],
                "step_index": metadata["record_contract"]["alignment"]["step_index_base"],
                "gold_token_id": None, "profile": None,
            }],
        }
        try:
            validate_atr_records([probe])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValidationError("ATR checkpoint training source provenance is invalid") from exc
    return rows


def _check_training_provenance(
    snapshot: list[dict[str, Any]], metadata: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    training = _validate_training_provenance(metadata)
    by_response = {row["response_id"]: row for row in training}
    cases = {row["case_id"] for row in training}
    steps = {step for row in training for step in row["step_ids"]}
    profiles = {row["profile_sha256"] for row in training}
    sources = {row["response_id"]: row for row in metadata["training_provenance"]["sources"]}
    current_sources = {record["response_id"]: {
        "response_id": record["response_id"], "source_kind": record["source_kind"],
        "provenance": record["provenance"],
    } for record in records}
    for row in snapshot:
        known_response = row["response_id"] in by_response
        overlap = (
            known_response or row["case_id"] in cases
            or bool(set(row["step_ids"]) & steps)
            or (row["profile_sha256"] is not None and row["profile_sha256"] in profiles)
        )
        if row["split"] != "train" and overlap:
            raise ValidationError(f"ATR training provenance overlap in non-train response {row['response_id']}")
        if row["split"] == "train":
            if known_response and (
                not _same_json_values(row, by_response[row["response_id"]])
                or not _same_json_values(current_sources[row["response_id"]], sources[row["response_id"]])
            ):
                raise ValidationError("ATR current train record differs from checkpoint training provenance")
            if not known_response and overlap:
                raise ValidationError("ATR new train response reuses a checkpoint training identity or profile")


def _validate_checkpoint_metadata(metadata: Any) -> ATRConfig:
    if not isinstance(metadata, dict) or metadata.get("checkpoint_version") != ATR_CHECKPOINT_VERSION:
        raise ValidationError("unsupported ATR checkpoint metadata")
    raw_config = metadata.get("config")
    if not isinstance(raw_config, dict) or set(raw_config) != set(ATRConfig.__dataclass_fields__):
        raise ValidationError("ATR checkpoint config fields are invalid")
    try:
        config = ATRConfig(**raw_config)
        config.validate()
    except (TypeError, ValueError) as exc:
        raise ValidationError("ATR checkpoint config is invalid") from exc
    if metadata.get("architecture") != config.architecture:
        raise ValidationError("ATR checkpoint architecture/config contract mismatch")
    shape = metadata.get("input_shape")
    if not isinstance(shape, list) or len(shape) != 3 or not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in shape
    ):
        raise ValidationError("ATR checkpoint input dimensions are invalid")
    contract = metadata.get("record_contract")
    _validate_record_contract(contract)
    feature_contract = contract["feature_contract"]
    if shape[1:] != [len(feature_contract["layer_ids"]), len(feature_contract["kv_head_ids"])]:
        raise ValidationError("ATR checkpoint spatial dimensions/feature contract mismatch")
    natural_width = metadata.get("train_max_natural_token_width")
    if not isinstance(natural_width, int) or isinstance(natural_width, bool) or natural_width <= 0:
        raise ValidationError("ATR checkpoint train token width is invalid")
    if (
        shape[0] != (config.token_width or natural_width) or natural_width > shape[0]
        or metadata.get("token_width_source") != (
            "explicit_config" if config.token_width else "maximum_over_train_split_only"
        )
    ):
        raise ValidationError("ATR checkpoint token width/config contract mismatch")
    candidates = validate_atr_vocabulary(metadata.get("candidate_vocabulary"), contract)
    if len(candidates) < 2:
        raise ValidationError("ATR checkpoint requires at least two train candidate tokens")
    if metadata.get("standard_resnet_width") != (config.base_channels == 64):
        raise ValidationError("ATR checkpoint model width/config contract mismatch")
    if not isinstance(metadata.get("run_kind"), str) or not metadata["run_kind"]:
        raise ValidationError("ATR checkpoint run kind is invalid")
    if metadata.get("scientific_result") is not False:
        raise ValidationError("ATR checkpoint must identify this reconstructed run as unvalidated")
    _validate_training_provenance(metadata)
    sources = metadata["training_provenance"]["sources"]
    source_kinds = sorted({row["source_kind"] for row in sources})
    synthetic = all(row["source_kind"] == "synthetic_fixture" for row in sources)
    if metadata.get("training_source_kinds") != source_kinds or metadata.get("synthetic_train_only") is not synthetic:
        raise ValidationError("ATR checkpoint source summary/provenance mismatch")
    if metadata["run_kind"] == "synthetic_smoke_only" and not synthetic:
        raise ValidationError("ATR synthetic smoke requires only synthetic training sources")
    return config


def _save_atr_checkpoint(
    weights_path: Path, metadata: dict[str, Any], state: dict[str, Tensor]
) -> None:
    _validate_checkpoint_metadata(metadata)
    _validate_qai_state_dict(state)
    if metadata.get("weights_file") != weights_path.name:
        raise ValidationError("ATR checkpoint weights filename mismatch")
    contract = _checkpoint_contract(metadata)
    digest = _contract_sha256(contract)
    torch.save({"contract": contract, "contract_sha256": digest, "state_dict": state}, weights_path)
    metadata["contract_sha256"] = digest
    metadata["weights_sha256"] = _sha256(weights_path)


def _coverage(
    records: list[dict[str, Any]], features: dict[tuple[str, str], Tensor], candidates: set[int]
) -> dict[str, Any]:
    steps = [(record["response_id"], step) for record in records for step in record["steps"]]
    gold = [(response_id, step) for response_id, step in steps if step["gold_token_id"] is not None]
    covered = sum(step["gold_token_id"] in candidates for _, step in gold)
    ce_steps = sum(
        step["gold_token_id"] in candidates and (response_id, step["step_id"]) in features
        for response_id, step in gold
    )
    return {
        "responses": len(records), "steps": len(steps), "gold_tokens": len(gold),
        "unlabeled_steps": len(steps) - len(gold),
        "candidate_gold_tokens": covered, "oov_gold_tokens": len(gold) - covered,
        "candidate_coverage": covered / len(gold) if gold else None,
        "available_profile_steps": sum((response_id, step["step_id"]) in features for response_id, step in steps),
        "cross_entropy_steps": ce_steps,
        "cross_entropy_gold_coverage": ce_steps / len(gold) if gold else None,
    }


def _evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, int, int]:
    model.eval()
    loss_total = 0.0
    correct = total = 0
    with torch.no_grad():
        for features, targets, *_ in loader:
            logits = model(features.to(device))
            _finite(logits, "validation logits")
            targets = targets.to(device)
            loss = nn.functional.cross_entropy(logits, targets)
            _finite(loss, "validation loss")
            loss_total += float(loss.item()) * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += len(targets)
    return loss_total / total if total else 0.0, correct, total


def train_atr(
    records: list[dict[str, Any]], config: ATRConfig, checkpoint_dir: str | Path,
    *, device: str = "auto", run_kind: str = "unvalidated_research_run",
) -> dict[str, Any]:
    config.validate()
    if not isinstance(run_kind, str) or not run_kind:
        raise ValidationError("ATR run_kind must be a non-empty string")
    records = validate_atr_records(records)
    contract = _run_contract(records)
    snapshot = atr_record_snapshot(records)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "validation"]
    if not train_records:
        raise ValidationError("ATR training requires train responses")
    if any(step["gold_token_id"] is None or step["profile"] is None for record in train_records for step in record["steps"]):
        raise ValidationError("ATR training requires a gold token and profile for every train step")
    if run_kind == "synthetic_smoke_only" and any(record["source_kind"] != "synthetic_fixture" for record in records):
        raise ValidationError("ATR synthetic smoke requires only synthetic records")
    vocabulary = freeze_atr_vocabulary(records)
    candidates = validate_atr_vocabulary(vocabulary, contract)
    if len(candidates) < 2:
        raise ValidationError("ATR training requires at least two train candidate tokens")
    token_to_index = {token: index for index, token in enumerate(vocabulary["token_ids"])}
    shape = derive_atr_input_shape(train_records, config)
    features = prepare_atr_features(train_records + validation_records, config, shape)
    for feature in features.values():
        _finite(feature, "prepared feature")
    train_dataset = ATRTokenDataset(train_records, features, token_to_index)
    if len(train_dataset) < 2:
        raise ValidationError("ATR BatchNorm training requires at least two train steps")
    validation_dataset = ATRTokenDataset(validation_records, features, token_to_index) if validation_records else None
    _seed_everything(config.seed)
    resolved_device = _device(device)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=_QAITrainingBatchSampler(len(train_dataset), config.batch_size, generator),
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers) if validation_dataset is not None and len(validation_dataset) else None
    model = QAIResNet18(shape[0], len(candidates), config.base_channels).to(resolved_device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    before = {key: value.detach().cpu().clone() for key, value in model.named_parameters()}
    history: list[dict[str, Any]] = []
    gradient_observed = False
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    started = time.monotonic()
    for epoch in range(config.epochs):
        model.train()
        loss_total = 0.0
        total = 0
        for inputs, targets, *_ in train_loader:
            inputs, targets = inputs.to(resolved_device), targets.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            _finite(logits, "training logits")
            loss = nn.functional.cross_entropy(logits, targets)
            _finite(loss, "training loss")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    _finite(parameter.grad, "training gradient")
                    gradient_observed = gradient_observed or bool(torch.count_nonzero(parameter.grad).item())
            optimizer.step()
            _validate_qai_state_dict(model.state_dict())
            total += len(targets)
            loss_total += float(loss.item()) * len(targets)
        result: dict[str, Any] = {"epoch": epoch + 1, "train_loss": loss_total / total, "train_total": total}
        if validation_loader is not None:
            val_loss, correct, val_total = _evaluate_loss(model, validation_loader, resolved_device)
            result.update(validation_loss=val_loss, validation_correct=correct, validation_total=val_total)
        history.append(result)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    duration = time.monotonic() - started
    changed = any(not torch.equal(before[key], value.detach().cpu()) for key, value in model.named_parameters())
    if not gradient_observed or not changed:
        raise ValidationError("ATR training did not observe a finite gradient and parameter update")
    if atr_record_snapshot(records) != snapshot:
        raise ValidationError("ATR records changed during training; no checkpoint is saved")
    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "model_state.pt"
    natural_width = max(len(step["profile"][0][0]) for record in train_records for step in record["steps"])
    metadata = {
        "checkpoint_version": ATR_CHECKPOINT_VERSION,
        "weights_file": weights_path.name, "architecture": config.architecture,
        "config": asdict(config), "input_shape": list(shape),
        "train_max_natural_token_width": natural_width,
        "token_width_source": "explicit_config" if config.token_width else "maximum_over_train_split_only",
        "candidate_vocabulary": vocabulary, "record_contract": contract,
        "training_provenance": _training_provenance(snapshot, records),
        "training_source_kinds": sorted({record["source_kind"] for record in train_records}),
        "synthetic_train_only": all(record["source_kind"] == "synthetic_fixture" for record in train_records),
        "model_parameter_count": parameter_count,
        "standard_resnet_width": config.base_channels == 64,
        "run_kind": run_kind, "scientific_result": False,
        "paper_backed": [
            "supervised per-decoding-step token prediction from reconstructed token sparsity",
            "fixed task-specific offline candidate vocabulary",
            "preceding-step aggregation without increasing feature dimension",
            "response-disjoint splitting and DASR over every gold token",
        ],
        "reconstruction_choices": [
            "convolutional ResNet18 layout and model width",
            "token-channel x layer x KV-head tensor layout",
            "causal running mean of preceding available normalized profiles with additive strength",
            "normalization toggle, train-only width padding and reject-longer policy",
            "optimizer and all numerical hyperparameters",
        ],
    }
    _save_atr_checkpoint(weights_path, metadata, {key: value.detach().cpu() for key, value in model.state_dict().items()})
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "checkpoint_dir": str(destination), "device": str(resolved_device),
        "duration_seconds": duration,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(resolved_device)) if resolved_device.type == "cuda" else 0,
        "train_responses": len(train_records), "train_steps": len(train_dataset),
        "validation_coverage": _coverage(validation_records, features, candidates),
        "model_parameter_count": parameter_count, "model_input_shape": list(shape),
        "standard_resnet_width": config.base_channels == 64,
        "token_width_source": metadata["token_width_source"],
        "candidate_token_ids": vocabulary["token_ids"],
        "candidate_vocabulary_sha256": vocabulary["sha256"],
        "gradient_observed": gradient_observed, "parameter_changed": changed,
        "history": history, "weights_sha256": metadata["weights_sha256"],
        "contract_sha256": metadata["contract_sha256"],
        "synthetic_train_only": metadata["synthetic_train_only"],
        "scientific_result": False,
    }


def load_atr_checkpoint(
    checkpoint_dir: str | Path, *, device: str = "auto"
) -> tuple[QAIResNet18, dict[str, Any], torch.device]:
    directory = Path(checkpoint_dir)
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid ATR checkpoint metadata") from exc
    config = _validate_checkpoint_metadata(metadata)
    contract = _checkpoint_contract(metadata)
    digest = _contract_sha256(contract)
    if metadata.get("contract_sha256") != digest:
        raise ValidationError("ATR checkpoint metadata contract digest mismatch")
    filename = metadata.get("weights_file")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValidationError("ATR checkpoint weights filename must be local")
    weights_path = directory / filename
    if not weights_path.is_file() or _sha256(weights_path) != metadata.get("weights_sha256"):
        raise ValidationError("ATR checkpoint weight digest mismatch")
    try:
        envelope = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValidationError("ATR safe state_dict loading failed") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"contract", "contract_sha256", "state_dict"}:
        raise ValidationError("ATR checkpoint requires a bound metadata/state contract")
    saved = envelope["contract"]
    if not isinstance(saved, dict) or _contract_sha256(saved) != digest or envelope["contract_sha256"] != digest:
        raise ValidationError("ATR checkpoint metadata/state contract mismatch")
    resolved_device = _device(device)
    model = QAIResNet18(metadata["input_shape"][0], len(metadata["candidate_vocabulary"]["token_ids"]), config.base_channels)
    if metadata.get("model_parameter_count") != sum(parameter.numel() for parameter in model.parameters()):
        raise ValidationError("ATR checkpoint parameter count/config contract mismatch")
    _validate_qai_state_dict(envelope["state_dict"], model.state_dict())
    try:
        model.load_state_dict(envelope["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValidationError("ATR checkpoint state/config contract mismatch") from exc
    model.to(resolved_device).eval()
    return model, metadata, resolved_device


def predict_atr(
    records: list[dict[str, Any]], checkpoint_dir: str | Path, *, split: str = "test",
    device: str = "auto", evaluate: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if split not in ATR_SPLITS:
        raise ValidationError(f"unsupported ATR inference split: {split}")
    if not isinstance(evaluate, bool):
        raise ValidationError("ATR evaluate must be boolean")
    records = validate_atr_records(records)
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise ValidationError(f"ATR inference split {split} has no responses")
    if evaluate and any(step["gold_token_id"] is None for record in selected for step in record["steps"]):
        raise ValidationError("ATR evaluation requires gold token IDs for every selected step")
    model, metadata, resolved_device = load_atr_checkpoint(checkpoint_dir, device=device)
    if not _same_json_values(_run_contract(records), metadata["record_contract"]):
        raise ValidationError("ATR inference task/tokenizer/alignment/feature contract does not match checkpoint")
    _check_training_provenance(atr_record_snapshot(records), metadata, records)
    config = ATRConfig(**metadata["config"])
    features = prepare_atr_features(selected, config, tuple(metadata["input_shape"]))
    candidates = metadata["candidate_vocabulary"]["token_ids"]
    predictions: list[dict[str, Any]] = []
    entries: list[tuple[int, Tensor]] = []
    for record in selected:
        for step in record["steps"]:
            key = (record["response_id"], step["step_id"])
            feature = features.get(key)
            prediction = {
                "response_id": record["response_id"], "case_id": record["case_id"],
                "split": record["split"], "task_id": record["task_id"],
                "step_id": step["step_id"], "step_index": step["step_index"],
                "predicted_token_id": None, "predicted_probability": None,
                "classification_skipped": "missing_profile" if feature is None else None,
            }
            predictions.append(prediction)
            if feature is not None:
                _finite(feature, "inference feature")
                entries.append((len(predictions) - 1, feature))
    with torch.no_grad():
        for start in range(0, len(entries), config.batch_size):
            batch = entries[start:start + config.batch_size]
            inputs = torch.stack([feature for _, feature in batch]).to(resolved_device)
            logits = model(inputs)
            _finite(logits, "inference logits")
            probabilities = torch.softmax(logits, dim=1).cpu()
            _finite(probabilities, "inference probabilities")
            for offset, candidate_index in enumerate(probabilities.argmax(dim=1).tolist()):
                prediction = predictions[batch[offset][0]]
                prediction["predicted_token_id"] = candidates[candidate_index]
                prediction["predicted_probability"] = float(probabilities[offset, candidate_index].item())
    metrics = compute_atr_dasr(selected, predictions, metadata["candidate_vocabulary"]) if evaluate else None
    return predictions, metrics


def create_synthetic_atr_fixture(root: str | Path, *, seed: int = 7) -> Path:
    """Twelve three-step responses with known/OOV/missing-profile test cases."""
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    contract = {
        "schema_version": "janus.atr.response.v1", "task_id": "synthetic-atr-task",
        "tokenizer": {"name": "synthetic-tokenizer", "revision": "fixture-v1", "vocab_size": 32},
        "alignment": {
            "step_index_base": 0, "profile_predicts": "same_index_output_token",
            "bos_included": False, "eos_included": False,
            "special_tokens": "excluded", "response_scope": "complete_response",
        },
        "feature_contract": {
            "data_stage": "reconstructed_token_sparsity",
            "axis_order": ["layer", "kv_head", "key_token"],
            "layer_ids": [f"layer-{index}" for index in range(4)],
            "kv_head_ids": [f"head-{index}" for index in range(4)],
            "key_position_policy": "absolute_zero_based_prefix_positions",
            "reconstruction": {"method": "synthetic-explicit", "revision": "fixture-v1", "parameters": {}},
        },
    }
    number = 0
    for split, count in (("train", 8), ("validation", 2), ("test", 2)):
        for response_offset in range(count):
            response_id = f"synthetic-{split}-response-{response_offset}"
            steps = []
            for step_index in range(3):
                token = (11, 17)[(response_offset + step_index) % 2]
                profile = rng.uniform(0.01, 0.06, size=(4, 4, 4)).astype(np.float32)
                profile[:, :, 0 if token == 11 else 3] += 1.0
                gold = token
                if split == "test" and response_offset == 0 and step_index == 1:
                    gold = 23  # Not present in any train step.
                payload = None if split == "test" and response_offset == 1 and step_index == 2 else profile.tolist()
                steps.append({
                    "step_id": f"{response_id}-step-{step_index}", "step_index": step_index,
                    "gold_token_id": gold, "profile": payload,
                })
            records.append({
                **contract, "response_id": response_id, "case_id": f"synthetic-case-{number}",
                "split": split, "source_kind": "synthetic_fixture", "provenance": {"synthetic": True},
                "steps": steps,
            })
            number += 1
    records = validate_atr_records(records, require_gold=True)
    manifest = destination / "manifest.jsonl"
    write_jsonl(manifest, records)
    return manifest


def run_synthetic_atr_smoke(root: str | Path, *, device: str = "auto") -> dict[str, Any]:
    destination = Path(root)
    records = load_atr_records(create_synthetic_atr_fixture(destination / "data"))
    config = ATRConfig(base_channels=2, batch_size=4, epochs=2, learning_rate=3e-3, seed=19)
    training = train_atr(records, config, destination / "checkpoint", device=device, run_kind="synthetic_smoke_only")
    predictions, metrics = predict_atr(records, destination / "checkpoint", device=device)
    if (
        len(predictions) != 6 or metrics is None
        or metrics["micro_denominator_all_gold_tokens"] != 6
        or metrics["out_of_vocabulary_gold_tokens_counted_incorrect"] != 1
        or metrics["missing_predictions"] != 1
        or not training["gradient_observed"] or not training["parameter_changed"]
    ):
        raise ValidationError("ATR synthetic smoke did not preserve the full token denominator and update contract")
    write_jsonl(destination / "predictions.jsonl", predictions)
    report = {
        "status": "ok", "synthetic_only": True, "scientific_result": False,
        "training": training, "test_predictions": len(predictions),
        "dasr_smoke_only": metrics, "checkpoint_reload_verified": True,
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report
