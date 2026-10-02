# Evidence-backed legacy dataset manifest

This milestone recovers one narrow join contract by static inspection. It does not
execute the legacy Python program and does not turn a selected top-10 output into an
unbiased evaluation set.

## Verified source set

- CSV: `prefill_attribute_inference/legal-llama/infer_10_val.csv`
- Payload pointer:
  `prefill_attribute_inference/legal-llama/infer_10_val/case_57/q_k_attn_rank_topk256.npz`
- Static source:
  `prefill_attribute_inference/legal-llama/load_model_infer_illness_top_10_load_npz_speed_up.py`
- Materialized payload: one checksum-matched copy of the pointer target, kept outside
  the Git checkout and not committed

The CSV has 10 rows and the columns `index`, `query`, `true_label`, `pred_label`, and
`pred_prob`. Its `index` values are unique, and index 57 occurs exactly once.

## Authority for the join

The statically inspected source supplies these relationships:

| Source lines | Evidence |
| --- | --- |
| 23–26 | Reads the CSV `index` and opens `case_{index}/q_k_attn_rank_topk256.npz`. |
| 188–202 | Copies the selected NPZ and emits the same `index`, `query`, `true_label`, `pred_label`, and `pred_prob` into the output CSV. |
| 208–209 | Declares and writes the five output CSV columns. |
| 79–83 | Uses validation-oriented variable names, which are retained only as naming evidence and not promoted to a canonical split. |

The manifest stores the full source-file SHA-256 and a SHA-256 for each cited line
range. The builder also requires the CSV row to be unique and the materialized NPZ
SHA-256 and size to match the checked-in Git LFS pointer.

## Deliberately unresolved fields

- `split` remains `unassigned`. Names such as `val`, `infer_10_val`, and `df_val` are
  useful provenance, but the CSV has no split column and no versioned split manifest
  was found.
- The CSV supplies one query-level attribute `true_label`; it supplies no token labels,
  response identifier, or token-alignment contract.
- `manifest_sample_id` is a new deterministic manifest identifier. It is not claimed
  to be an original collection identifier.
- The included prediction is a selected legacy output, not a controlled rerun joined
  by the new manifest identifier.
- The NPZ is a reconstructed prefill rank tensor, not a raw L2/TLB probe trace.

Consequently this sample remains ineligible for PASR and DASR reporting. To make a
real evaluation eligible, the minimum additional materials are: a versioned
non-selected split manifest keyed by stable original sample IDs; complete predictions
keyed by those IDs; and, for DASR, response IDs plus gold token sequences and their
tokenization/alignment contract.
