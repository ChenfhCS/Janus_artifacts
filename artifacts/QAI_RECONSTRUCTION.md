# QAI learning pipeline: evidence and reconstruction choices

This module is a runnable reconstruction of the Query Attribute Inference (QAI)
learning stage. It is not a reproduction of the paper's reported PASR.

## What the paper specifies

The main paper states that QAI is supervised classification over reconstructed
token-level prefill sparsity profiles (Section 6.1, pages 8-9). It further specifies:

- normalization removes global scale differences caused by query length and total
  attention activity;
- signals from all monitored layers and heads are retained and composed instead of
  being collapsed prematurely;
- the primary profiling setup uses 5,000 QAI queries per model, an 8:1:1
  sample-disjoint split, and task-specific 18- or 34-layer residual predictors
  (Section 7.4, page 11);
- PASR counts correct attribute predictions among queries containing that attribute
  (Section 7.1, page 9).

The supplement discusses updating prediction models under model evolution but does
not supply optimizer, learning-rate, epoch, batch-size, or checkpoint-format details.

## Static artifact evidence

The public QAI scripts were read as text and were never imported or executed. Across
the inspected scripts, the legacy preprocessing:

1. reads `attn_rank` as `(layer, head, query-token, key-token)`;
2. selects a configurable number of the largest rank entries per query token;
3. counts selected key-token positions for every layer/head;
4. applies min-max normalization independently to each layer/head profile; and
5. transposes the result to `(key-token, layer, head)` before inference.

This establishes what the checked-in artifact does, not the semantic polarity of the
rank values. The paper does not define whether larger or smaller stored integers denote
stronger attention. This implementation therefore names the choice
`rank_selection=largest_values`, records it in every feature/checkpoint, and requires
author confirmation before treating it as a faithful scientific reconstruction.

The checkpoint names include `resnet18`, supporting the choice of the 18-layer branch.
The legacy `.pt` files are not loaded because the scripts use whole-object pickle
loading and do not provide a safe, portable architecture contract.

## Implemented path

`janus_artifact/qai.py` provides:

- strict ID-keyed JSONL records with case-group split isolation and a policy rejecting identical NPZ bytes across splits;
- NPZ loading through the existing `allow_pickle=False` reader;
- rank-histogram features and per-layer/head min-max normalization;
- an explicit axis contract: input `(layer, head, query-token, key-token)`, histogram
  reduction over query-token and selected-rank-slot, and output
  `(key-token-channel, layer-height, head-width)`;
- a locally implemented ResNet-18 with configurable input channels;
- deterministic AdamW training, validation, inference, and PASR generation;
- v2 checkpoints containing only tensor state and primitive JSON contract values,
  loaded with `torch.load(..., weights_only=True)`, with matching metadata and
  contract digests; and
- a synthetic-only smoke test covering gradients, parameter updates, save/load,
  inference, and PASR plumbing.

## Explicit reconstruction choices

The paper does not provide the following values. The checked-in configuration records
them as reconstruction choices rather than original Janus parameters:

| Choice | Default |
| --- | --- |
| Architecture branch | ResNet-18 |
| Rank entries retained per query token | 10 |
| Rank interpretation | largest stored values, matching static artifact code only |
| Feature layout | token positions as channels, layer x head as the 2-D grid |
| Normalization | independent min-max per layer/head |
| Normalization fit scope | per sample; no train/test dataset statistics |
| Variable token width | pad to explicit width or train-only maximum; reject longer non-train inputs |
| Base residual width | 64 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Epochs | 20 |
| Seed | 1337 |

Base width 64 is the standard ResNet width. For an inspected 278-channel input and
four output classes, the untrained full-width model has 12,040,964 parameters; the
class count is taken only from a legacy filename and is not treated as a verified
dataset contract. The synthetic smoke deliberately narrows the base width to 8. Its
16-channel, two-class model has 181,938 parameters, uses batch size 4 and two epochs,
and is only a plumbing check.

Token width is never derived from validation or test records. With `token_width=0`,
the trainer takes the maximum natural width among present-attribute training records,
zero-pads shorter samples on the key-token axis, and rejects any longer validation or
test sample instead of truncating it. Layer and head dimensions must match exactly;
the code does not infer a layer/head alignment.

## Required record format

Each JSONL record has exactly:

```json
{"sample_id":"...","case_id":"...","split":"train|validation|test","attribute":"...","attribute_present":true,"label":"...","npz_path":"..."}
```

For an absent attribute, `attribute_present` is `false` and `label` must be `null`.
Absent records do not enter the classifier and remain outside the PASR denominator;
the current module does not claim to solve attribute-presence detection. All records
in one run must target one attribute. The fine-class label vocabulary is frozen from
present training records, and unknown validation/test labels are rejected. Every
`case_id` must stay in exactly one split.

## Minimum author material for real evaluation

For QAI, the minimum missing material is a non-selected manifest with stable original
sample/case IDs, attribute-presence and gold-label fields, immutable 8:1:1 assignments,
payload checksums, monitored layer/head ordering, token-width/padding policy, and all
held-out predictions keyed by sample ID.

For ATR, the minimum missing material is response-level split assignments, per-step
trace-to-response IDs, gold token IDs with tokenizer name/revision and alignment rules,
the train-only candidate vocabulary with OOV policy, the sequential-augmentation
definition, and all test predictions keyed by response and decoding step.


## Audit repair contracts

Training retains every original train sample ID, case ID, attribute, presence/label
value, and present-attribute payload SHA-256 in the checkpoint contract. Absent
records are excluded from classification but their sample/case identities remain
in provenance; their unused payloads are not opened. Before training, the complete
manifest is checked for duplicate sample IDs, case split leakage, and identical NPZ
bytes across splits. Hashing held-out files is an isolation check and does not fit
features or model parameters. Payload snapshots are checked again before saving.

Inference checks the complete supplied manifest against checkpoint training
provenance, including splits other than the requested inference split. A non-train
record sharing any train sample ID, case ID, or present payload digest is rejected.
Known train sample IDs must retain their recorded semantics. This permits an
independent disjoint evaluation manifest and blocks reassignment of training cases
or identical bytes under new IDs. The payload policy compares exact NPZ bytes;
different encodings of the same array are not detected by this byte digest.
All supplied records, including mixed present/absent and absent-only inputs, must
target the checkpoint attribute before PASR is computed.

The v2 checkpoint stores the same canonical semantic contract in the JSON sidecar
and the safe tensor-state envelope. Labels must be unique and sorted; config,
architecture, feature dimensions, token-width policy, model tensor keys/shapes/
dtypes, and training provenance are checked. Non-finite state, gradients, losses,
logits, and probabilities are rejected. SHA-256 checks detect corruption and
metadata/weights mismatch; they do not authenticate an attacker who can rewrite
both files and recompute their checksums. Old unbound v1 checkpoints are rejected
and require retraining; there is no unsafe pickle compatibility fallback.

Training reshuffles all sample indices deterministically each epoch. If the final
batch would have one sample, it is merged into the preceding batch, so the largest
batch can contain `batch_size + 1` samples. No sample is dropped. Batch size one is
rejected before payload reads. The regression with 33 samples, batch size 32, and
a 32-by-32 layer/head grid reaches a one-by-one final spatial grid with batch 33
and retains all 33 samples.

The NPZ adapter checks exact unique archive members and compressed-file/
uncompressed-directory budgets before decompression or CRC traversal. It then
reads bounded NPY headers and validates shape, integer dtype, and declared array
bytes against the member body before any NumPy data allocation. CRC validation
and `allow_pickle=False` loading run only after these checks pass.
