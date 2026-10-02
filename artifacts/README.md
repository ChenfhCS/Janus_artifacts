# Janus artifact refactor: controlled offline pipeline

This directory adds a standard-library validation, split, vocabulary, replay, and
metric layer around the existing Janus artifact, plus optional PyTorch QAI and ATR
learning pipelines. The optional legacy NPZ adapter uses NumPy with pickle loading disabled; NumPy is already present in the initialized
project environment. The refactor does not modify or execute legacy attack scripts
and does not claim to reproduce the GPU side channel.

The paper describes: (1) phase-aware SIMA extraction, (2) page-to-token sparsity
reconstruction, and (3) query attribute inference / autoregressive token recovery.
`MODULE_MAP.md` maps those stages to the public artifact; `GAPS.md` records what is
still unavailable; `TRACE_SCHEMA.md` defines provenance-safe JSONL records; and
`REAL_TRACE_VALIDATION.md` records the one safely inspected legacy NPZ.
`DATASET_MANIFEST.md` records the recovered CSV-index join and its remaining gaps.
`QAI_RECONSTRUCTION.md` distinguishes the paper-backed QAI design from explicit
reconstruction choices in the new runnable learning pipeline.
[ATR_RECONSTRUCTION.md](ATR_RECONSTRUCTION.md) defines the executable token-learning
contract, causal augmentation choices and real-data gaps;
[ATR_VALIDATION.md](ATR_VALIDATION.md) records actual synthetic software checks.
[UPSTREAM_RECONSTRUCTION.md](UPSTREAM_RECONSTRUCTION.md) defines the controlled
toy workload, raw probe contract, reference phase alignment and representative
page-to-token operators; [UPSTREAM_VALIDATION.md](UPSTREAM_VALIDATION.md) records
the connected CPU replay/QAI/ATR check. Probe observations are simulated, and
phase boundaries and gold labels remain explicit oracle annotations.

[OWNED_GPU_TIMING.md](OWNED_GPU_TIMING.md) describes a separate bounded CUDA
timing and exact CPU ground-truth benchmark using two owned workers. It has no
cache/TLB collector or model workload; actual cross-context GPU overlap stays
unknown without a shared GPU timebase or profiler evidence.
[OWNED_GPU_TIMING_VALIDATION.md](OWNED_GPU_TIMING_VALIDATION.md) records the
actual full regression and the preserved failed post-idle hardware result.
[SELECTOR_REPLAY.md](SELECTOR_REPLAY.md) defines the replaceable offline selector,
static KV cache, selected-row attention and separate probe request boundary.
It has an executable synthetic CPU closure; real model and probe integration
remain pending.
[SELECTOR_REPLAY_VALIDATION.md](SELECTOR_REPLAY_VALIDATION.md) records the final
CPU regression and actual CLI closure.


## Run the self-contained checks

From the initialized project directory, activate its environment and enter the
repository checkout:

```bash
source scripts/activate.sh
cd repo
python -m unittest discover -s artifacts/tests -v
```

```bash
python -m artifacts.janus_artifact.cli validate-traces \
  --input artifacts/fixtures/synthetic_traces.jsonl

python -m artifacts.janus_artifact.cli adapt-legacy-manifest \
  --input artifacts/fixtures/legacy_lfs_manifest.jsonl \
  --output /tmp/janus-adapted-pointers.jsonl

python -m artifacts.janus_artifact.cli adapt-prefill-rank-npz \
  --input /path/to/verified/q_k_attn_rank_topk256.npz \
  --legacy-path prefill_attribute_inference/legal-llama/infer_10_val/case_57/q_k_attn_rank_topk256.npz \
  --case-id case-57 \
  --output /tmp/janus-adapted-real.jsonl

# Build a one-sample manifest from a CSV, statically inspected source, LFS pointer,
# and checksum-matched materialized payload. See DATASET_MANIFEST.md for the exact
# paths used in the verified example.
python -m artifacts.janus_artifact.cli build-prefill-dataset-manifest \
  --dataset-id legal-llama-infer-10-val \
  --csv /path/to/infer_10_val.csv \
  --csv-legacy-path prefill_attribute_inference/legal-llama/infer_10_val.csv \
  --script /path/to/load_model_infer_illness_top_10_load_npz_speed_up.py \
  --script-legacy-path prefill_attribute_inference/legal-llama/load_model_infer_illness_top_10_load_npz_speed_up.py \
  --pointer /path/to/checked-out/case_57/q_k_attn_rank_topk256.npz \
  --payload /path/to/materialized/q_k_attn_rank_topk256.npz \
  --payload-legacy-path prefill_attribute_inference/legal-llama/infer_10_val/case_57/q_k_attn_rank_topk256.npz \
  --output /tmp/janus-dataset-manifest.json

python -m artifacts.janus_artifact.cli replay \
  --input artifacts/fixtures/synthetic_traces.jsonl \
  --output /tmp/janus-replay.jsonl

# A record using external_npz storage must be replayed with the same payload:
python -m artifacts.janus_artifact.cli replay \
  --input /tmp/janus-adapted-real.jsonl \
  --external-payload /path/to/verified/q_k_attn_rank_topk256.npz \
  --output /tmp/janus-real-replay.jsonl

python -m artifacts.janus_artifact.cli freeze-vocabulary \
  --input artifacts/fixtures/synthetic_traces.jsonl \
  --output /tmp/janus-vocabulary.json

python -m artifacts.janus_artifact.cli metrics \
  --pasr artifacts/fixtures/pasr_records.jsonl \
  --dasr artifacts/fixtures/dasr_records.jsonl \
  --vocabulary /tmp/janus-vocabulary.json

# Safe feature-only check: no label, model, or evaluation is inferred.
python -m artifacts.janus_artifact.cli qai-inspect-feature \
  --input /path/to/verified/q_k_attn_rank_topk256.npz \
  --selected-ranks 10

# Tiny synthetic-only gradient/checkpoint/inference smoke tests.
python -m artifacts.janus_artifact.cli qai-synthetic-smoke \
  --work-dir /tmp/janus-qai-smoke \
  --device auto

python -m artifacts.janus_artifact.cli atr-synthetic-smoke \
  --work-dir /tmp/janus-atr-smoke \
  --device cpu

# Controlled toy operation -> simulated probes -> profiles -> replay/QAI/ATR.
# The destination must be new; existing files are preserved.
python -m artifacts.janus_artifact.cli upstream-synthetic-smoke \
  --work-dir /tmp/janus-upstream-smoke \
  --device cpu
```

## Metric denominators

- PASR: for each attribute, correct predictions divided by queries containing that
  attribute. A missing prediction is incorrect; a query without that attribute is
  outside that attribute's denominator.
- DASR: token correctness per response, averaged across responses. Every gold test
  token remains in the denominator. Missing predictions and tokens outside a frozen
  candidate vocabulary are incorrect. Extra predictions are reported and ignored.

Vocabulary construction accepts only `train` records and emits a digest-protected,
frozen token list. Grouped splitting keeps every original `case_id` in exactly one
split. These contracts prevent row-order alignment and test-set vocabulary leakage.

The dataset-manifest join is also ID-based: the numeric CSV `index` must uniquely
match the `case_{index}` directory. CSV row order is never used.


The QAI audit repair uses v2 checkpoints with required training provenance,
cross-manifest sample/case/payload isolation, strict checkpoint attribute matching,
and a metadata contract bound to the safe tensor state. Batch size one is rejected;
a singleton final training batch is merged without dropping samples. NPZ member
budgets and NPY declarations are checked before CRC traversal and array allocation.
See [QAI_RECONSTRUCTION.md](QAI_RECONSTRUCTION.md) for exact policies and limits.

ATR requires explicit response/step identities, tokenizer revision, alignment,
gold token IDs for evaluation, and declared reconstructed profile axes. Its train
vocabulary and feature width are frozen before inference. Missing and OOV test
tokens remain in the full DASR denominator. Pure prediction uses --no-evaluate
and emits not_evaluated. These software checks do not reproduce paper attack rates.
