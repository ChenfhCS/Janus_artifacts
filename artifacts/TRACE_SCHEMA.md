# Janus trace contract (v1)

This schema records provenance before values. A consumer must never infer that a
record came from a real GPU collection merely because it contains numeric arrays.

Required top-level fields:

- `schema_version`: exactly `janus.trace.v1`.
- `trace_id`: globally unique within an input file.
- `case_id`: original-case group used for split isolation.
- `response_id`: required for decoding records and used for response-level evaluation.
- `split`: `unassigned`, `train`, `validation`, or `test`.
- `phase`: `prefill` or `decoding`.
- `source_kind`: one of the provenance classes below.
- `provenance`: source-specific evidence.
- `trace`: phase-aware observations, or `null` for a metadata-only LFS pointer.
- `labels`: optional evaluation-only attribute and token labels.

## Provenance classes

| `source_kind` | Meaning | Required evidence |
| --- | --- | --- |
| `real_collection` | Observed from an actual side-channel collection run | `collection_run_id`; trace payload |
| `refactored_legacy` | Adapted from an existing artifact without changing its scientific origin | `legacy_path`; payload or explicit `lfs_pointer_only` status |
| `oracle_annotation` | Ground truth or labels obtained through instrumentation/white-box access | `oracle_method`; never presented as an observable side channel |
| `replay_derived` | Deterministic output derived from recorded inputs | non-empty `parent_trace_ids`; replay digest |
| `synthetic_fixture` | Hand-authored test data | `synthetic: true`; never used for scientific claims |

## Phase-aware observation shapes

- Prefill is cumulative: `observations` is one non-empty numeric vector.
- Decoding is step-wise: `observations` is a non-empty matrix with one equal-width
  numeric vector per decoding step.
- `granularity` is explicit (`page` or `token`); replay does not silently promote
  page observations to token-level reconstruction.
- Inline records use `storage: inline` (the default) and embed observations.
- Safely inspected legacy prefill payloads use `storage: external_npz`,
  `data_stage: reconstructed_sparsity`, and `aggregation: rank_tensor`. They store
  only verified file/array metadata in JSONL; replay requires the caller to supply
  the payload again and revalidates its SHA-256, size, member list, shapes, dtypes,
  and array digests.

`labels` are evaluation-only. They must not be treated as data available to the
online attacker. This first milestone does not implement a collector, an oracle,
or the paper's probabilistic reconstruction algorithms.
