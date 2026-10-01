"""Synthetic ATR records for contract and feature regressions."""

from copy import deepcopy


def synthetic_response(response_id="response-a", split="train", count=3, offset=0.0):
    return {
        "schema_version": "janus.atr.response.v1",
        "response_id": response_id,
        "case_id": f"case-{response_id}",
        "task_id": "synthetic-atr-task",
        "split": split,
        "source_kind": "synthetic_fixture",
        "provenance": {"synthetic": True},
        "tokenizer": {"name": "synthetic-tokenizer", "revision": "synthetic-v1", "vocab_size": 32},
        "alignment": {
            "step_index_base": 0,
            "profile_predicts": "same_index_output_token",
            "bos_included": False,
            "eos_included": True,
            "special_tokens": "included",
            "response_scope": "complete_response",
        },
        "feature_contract": {
            "data_stage": "reconstructed_token_sparsity",
            "axis_order": ["layer", "kv_head", "key_token"],
            "layer_ids": ["layer-0", "layer-1"],
            "kv_head_ids": ["kv-head-0", "kv-head-1"],
            "key_position_policy": "absolute_zero_based_prefix_positions",
            "reconstruction": {
                "method": "synthetic_token_profile", "revision": "v1", "parameters": {}
            },
        },
        "steps": [
            {
                "step_id": f"{response_id}-step-{index}",
                "step_index": index,
                "gold_token_id": 11 if index % 2 == 0 else 17,
                "profile": [
                    [
                        [offset + index + layer + head + 1.0,
                         offset + index + layer + head + 2.0,
                         offset + index + layer + head + 3.0]
                        for head in range(2)
                    ]
                    for layer in range(2)
                ],
            }
            for index in range(count)
        ],
    }


def with_profile(record, profile):
    result = deepcopy(record)
    for step in result["steps"]:
        step["profile"] = deepcopy(profile)
    return result
