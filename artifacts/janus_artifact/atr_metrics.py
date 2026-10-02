"""Step-ID aligned ATR scoring with every gold token retained."""

from __future__ import annotations

import math
from typing import Any, Iterable

from .atr_data import atr_record_contract, validate_atr_records, validate_atr_vocabulary
from .metrics import compute_dasr
from .schema import ValidationError


def compute_atr_dasr(
    records: list[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    vocabulary: dict[str, Any],
) -> dict[str, Any]:
    records = validate_atr_records(records, require_gold=True)
    candidates = validate_atr_vocabulary(vocabulary, contract=atr_record_contract(records[0]))
    steps = {
        (record["response_id"], step["step_id"]): (record, step)
        for record in records for step in record["steps"]
    }
    aligned: dict[tuple[str, str], int | None] = {}
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValidationError("ATR prediction must be an object")
        response_id = prediction.get("response_id")
        step_id = prediction.get("step_id")
        if not isinstance(response_id, str) or not isinstance(step_id, str):
            raise ValidationError("ATR prediction requires response_id and step_id")
        key = (response_id, step_id)
        if key not in steps:
            raise ValidationError("ATR prediction refers to an unknown response/step")
        if key in aligned:
            raise ValidationError("duplicate ATR step prediction")
        record, step = steps[key]
        for field in ("case_id", "split", "task_id"):
            if field in prediction and prediction[field] != record[field]:
                raise ValidationError(f"ATR prediction {field} does not match the response")
        if "step_index" in prediction and (
            type(prediction["step_index"]) is not int
            or prediction["step_index"] != step["step_index"]
        ):
            raise ValidationError("ATR prediction step_index does not match its step_id")
        token = prediction.get("predicted_token_id")
        if token is not None and (type(token) is not int or token not in candidates):
            raise ValidationError("ATR predicted token must belong to the frozen train vocabulary")
        probability = prediction.get("predicted_probability")
        if probability is not None and (
            type(probability) not in (int, float)
            or not 0 <= probability <= 1 or not math.isfinite(probability)
        ):
            raise ValidationError("ATR prediction probability must be finite and in [0, 1]")
        aligned[key] = token

    rows = []
    for record in records:
        rows.append({
            "response_id": record["response_id"],
            "gold_tokens": [str(step["gold_token_id"]) for step in record["steps"]],
            "predicted_tokens": [
                None if aligned.get((record["response_id"], step["step_id"])) is None
                else str(aligned[(record["response_id"], step["step_id"])])
                for step in record["steps"]
            ],
        })
    result = compute_dasr(rows, frozen_vocabulary={str(token) for token in candidates})
    denominator = result["micro_denominator_all_gold_tokens"]
    covered = denominator - result["out_of_vocabulary_gold_tokens_counted_incorrect"]
    result.update({
        "step_alignment": "explicit_response_id_and_step_id_to_zero_based_step_index",
        "candidate_coverage": covered / denominator,
        "candidate_covered_gold_tokens": covered,
        "evaluation_kind": (
            "synthetic_smoke_only" if all(r["source_kind"] == "synthetic_fixture" for r in records)
            else "recorded_profiles_unvalidated"
        ),
        "scientific_result": False,
    })
    return result
