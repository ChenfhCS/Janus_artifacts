# Verified legacy trace sample

One public Git LFS payload was materialized outside the repository working tree
and inspected with `numpy.load(..., allow_pickle=False)`. The payload is not
committed by this refactor.

## Source and integrity

- Legacy path: `prefill_attribute_inference/legal-llama/infer_10_val/case_57/q_k_attn_rank_topk256.npz`
- LFS size: 1,060,689 bytes
- Verified payload SHA-256: `862d23b2b4bb661c8d8e435b61e959979b9fde4511b634be638c2b9d7b080cda`
- ZIP CRC: valid

## Structure

| Key | dtype | shape | observed range |
| --- | --- | --- | --- |
| `attn_rank` | `uint16` | `(32, 32, 3, 278)` | `0..256` |
| `top_k` | `uint16` | scalar | `256` |

The repository describes these files as reconstructed sparsity-pattern inputs,
and the legacy inference script consumes `attn_rank` as a four-dimensional
layer/head/query/key tensor. The adapter therefore records it as
`refactored_legacy`, `reconstructed_sparsity`, `token`, and `rank_tensor`.
It explicitly records `raw_probe_trace: false`.

The sibling CSV contains exactly one row whose explicit `index` equals `57`, but
the public artifact supplies no versioned join contract, split manifest, or stable
prediction/ground-truth record IDs. The adapter consequently leaves `split` as
`unassigned`, leaves `labels` empty, and declares both PASR and DASR ineligible.
It does not infer scientific labels or token alignment from directory names or row
position.
