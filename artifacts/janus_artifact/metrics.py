"""Paper-aligned PASR and DASR counting with explicit denominators."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .schema import ValidationError


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_pasr(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute PASR for each attribute over queries containing that attribute."""
    seen: set[tuple[str, str]] = set()
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "denominator_present_queries": 0, "unanswered": 0}
    )
    for row in rows:
        record_id = row.get("record_id")
        attribute = row.get("attribute")
        if not isinstance(record_id, str) or not record_id:
            raise ValidationError("PASR record_id must be a non-empty string")
        if not isinstance(attribute, str) or not attribute:
            raise ValidationError("PASR attribute must be a non-empty string")
        key = (record_id, attribute)
        if key in seen:
            raise ValidationError(f"duplicate PASR record: {record_id}/{attribute}")
        seen.add(key)
        present = row.get("attribute_present")
        if not isinstance(present, bool):
            raise ValidationError("PASR attribute_present must be boolean")
        if not present:
            continue
        gold = row.get("gold_value")
        if gold is None:
            raise ValidationError("present PASR attributes require gold_value")
        predicted = row.get("predicted_value")
        bucket = counts[attribute]
        bucket["denominator_present_queries"] += 1
        if predicted is None:
            bucket["unanswered"] += 1
        elif predicted == gold:
            bucket["correct"] += 1

    per_attribute: dict[str, dict[str, Any]] = {}
    for attribute, bucket in sorted(counts.items()):
        per_attribute[attribute] = {
            **bucket,
            "pasr": _rate(bucket["correct"], bucket["denominator_present_queries"]),
        }
    denominators = [item for item in per_attribute.values() if item["pasr"] is not None]
    macro = (
        sum(item["pasr"] for item in denominators) / len(denominators)
        if denominators
        else None
    )
    total_correct = sum(item["correct"] for item in per_attribute.values())
    total_denominator = sum(
        item["denominator_present_queries"] for item in per_attribute.values()
    )
    return {
        "metric": "PASR",
        "per_attribute": per_attribute,
        "macro_pasr": macro,
        "micro_correct": total_correct,
        "micro_denominator_present_queries": total_denominator,
        "micro_pasr": _rate(total_correct, total_denominator),
    }


def compute_dasr(
    rows: Iterable[dict[str, Any]], *, frozen_vocabulary: set[str] | None = None
) -> dict[str, Any]:
    """Average per-response token recovery rates over every gold test token."""
    seen: set[str] = set()
    responses: list[dict[str, Any]] = []
    total_correct = 0
    total_tokens = 0
    total_missing = 0
    total_oov = 0
    total_extra = 0
    for row in rows:
        response_id = row.get("response_id")
        if not isinstance(response_id, str) or not response_id:
            raise ValidationError("DASR response_id must be a non-empty string")
        if response_id in seen:
            raise ValidationError(f"duplicate DASR response_id: {response_id}")
        seen.add(response_id)
        gold = row.get("gold_tokens")
        predicted = row.get("predicted_tokens", [])
        if not isinstance(gold, list) or not gold or not all(
            isinstance(token, str) for token in gold
        ):
            raise ValidationError("DASR gold_tokens must be a non-empty string array")
        if not isinstance(predicted, list) or not all(
            isinstance(token, str) or token is None for token in predicted
        ):
            raise ValidationError("DASR predicted_tokens must be a string/null array")
        correct = 0
        missing = max(0, len(gold) - len(predicted)) + sum(
            token is None for token in predicted[:len(gold)]
        )
        extra = max(0, len(predicted) - len(gold))
        oov = 0
        for index, gold_token in enumerate(gold):
            in_vocabulary = frozen_vocabulary is None or gold_token in frozen_vocabulary
            if not in_vocabulary:
                oov += 1
                continue
            if index < len(predicted) and predicted[index] == gold_token:
                correct += 1
        denominator = len(gold)
        responses.append(
            {
                "response_id": response_id,
                "correct": correct,
                "denominator_all_gold_tokens": denominator,
                "missing_predictions": missing,
                "extra_predictions_ignored": extra,
                "out_of_vocabulary_gold_tokens_counted_incorrect": oov,
                "response_dasr": correct / denominator,
            }
        )
        total_correct += correct
        total_tokens += denominator
        total_missing += missing
        total_oov += oov
        total_extra += extra
    if not responses:
        raise ValidationError("DASR requires at least one response")
    return {
        "metric": "DASR",
        "definition": "mean per-response correctness; every gold test token is retained",
        "responses": responses,
        "dasr_macro": sum(item["response_dasr"] for item in responses) / len(responses),
        "micro_correct": total_correct,
        "micro_denominator_all_gold_tokens": total_tokens,
        "micro_rate": total_correct / total_tokens,
        "missing_predictions": total_missing,
        "out_of_vocabulary_gold_tokens_counted_incorrect": total_oov,
        "extra_predictions_ignored": total_extra,
    }
