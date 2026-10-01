# ATR reconstruction: executable offline learning contract

This module trains and applies a task-specific token classifier to already
reconstructed decoding-step sparsity profiles. It does not collect SIMA traces,
reconstruct page traces, run a victim model, or reproduce published Janus DASR.
Every output report records scientific_result=false.

## Paper basis and explicit choices

References: Janus_EuroSys_27(1).pdf (Library
libfile_1db0e1864f488191b9f18508195115a2, version 0) and the verified
Janus_EuroSys_27_Supplement(1).pdf (Library
libfile_623cf31c51d481918c3c727426d0946b, version 0). Page numbers are PDF pages.
The PDFs are evidence references, not source-package material.

| Contract | Paper evidence | Implementation choice |
| --- | --- | --- |
| Token classification | Main p9, section 6.2: decoding-step reconstructed token sparsity corresponds to output tokens; monitored layers and KV heads are combined | Explicit response/step IDs and gold token IDs; no row-order or text-token guessing |
| Predictor | Main p6 mentions MLP classifiers; main p11 describes task-specific 18-/34-layer residual predictors | Reuse clean QAI ResNet18; its convolutions, width and layout are reconstruction choices, not an exact author architecture |
| Sequential augmentation | Main p9, section 6.2: aggregate preceding-step information without increasing dimensionality | Causal additive running mean; formula and coefficient are not supplied by the paper |
| Candidate set | Main p9, section 6.2; supplement p2, section B: offline task-specific vocabulary is fixed before online/test use | Sorted unique gold token IDs from train only, frozen and checkpoint-bound |
| Split | Main p11: 8:1:1 sample-disjoint; supplement p2, section B: common response-level split | Optional 8:1:1 case-grouped split is stronger than response isolation; all steps stay together |
| DASR | Main p9, section 7.1: mean response-level token accuracy; supplement p2, section B: keep all test tokens, count out-of-set tokens wrong | Macro average of complete responses; OOV and missing predictions stay incorrect in the full gold denominator |

The supplement p3 identifies decoding-step alignment as affecting DASR. The
papers do not specify exact tokenization, BOS/EOS treatment, candidate selection,
feature layout, optimizer or augmentation formula.

## Required input

atr_data.py accepts JSON arrays, a single response object, or JSONL. Every
response has schema_version=janus.atr.response.v1 and exactly these fields:

| Field | Required meaning |
| --- | --- |
| response_id, case_id, task_id | Stable nonempty identities; response/step IDs are globally unique in a manifest; one case cannot cross splits |
| split | train, validation or test |
| source_kind, provenance | synthetic_fixture with synthetic=true, or reconstructed_recorded with explicit collection_run_id, profile_source_sha256, split_assignment_id and alignment_evidence_id |
| tokenizer | name, revision and positive vocab_size; gold IDs are integers in its range |
| alignment | step_index_base=0, profile_predicts=same_index_output_token, response_scope=complete_response, boolean bos_included/eos_included and special_tokens=included/excluded |
| feature_contract | data_stage=reconstructed_token_sparsity; axis_order=[layer,kv_head,key_token]; ordered unique layer_ids/kv_head_ids; key_position_policy=absolute_zero_based_prefix_positions |
| feature_contract.reconstruction | method, revision and primitive JSON parameters |
| steps | Complete nonempty sequence of step_id, contiguous zero-based step_index, gold_token_id and profile |

Profiles are finite nonnegative rectangular layer x KV-head x key-token arrays.
Key-token width may vary by step; layer/head order is fixed. Missing observations
are profile=null without removing the step or shifting later labels. Every train
step requires a profile and gold ID. Gold may be null for pure prediction; every
selected step needs gold for evaluation. Excluding special tokens while declaring
BOS/EOS included is rejected.

Recorded inputs require explicit non-placeholder tokenizer/reconstruction
revisions. Missing task, tokenizer, alignment, evidence, order or required labels
is an error. The pipeline never guesses them from selected CSV rows, filenames
or legacy scripts. A valid declaration is not proof of author provenance.
Tokenizers and models are not downloaded.

All responses must agree on task, tokenizer, alignment and feature contract.
Canonical JSON comparisons preserve value types across checkpoint contracts and
training source evidence: true, 1 and 1.0 differ; object key order does not.
Unknown fields, duplicate JSON keys, non-finite values and malformed Unicode are
rejected. Files are capped at 64 MiB and profile elements at eight million before
copying. Values that overflow float32 are rejected.

## Features and causal augmentation

Model input is float32 with shape (token_channels, monitored_layers,
monitored_KV_heads), transposed from the declared axes. This is not a paper-defined
grid. token_width=0 uses maximum observed train width; an explicit width can
override it. Shorter profiles get zero padding on the absolute key-position tail.
Longer profiles are rejected; validation/test never change the train shape.

normalization=none preserves values. Optional per_layer_head_minmax normalizes
each observed layer/head along its natural key-token axis before padding;
constant vectors become zero. No validation/test statistics are fitted.

Let f_t be the unaugmented normalized/padded tensor at step t, and H_t the earlier
steps of that same response with observed profiles:

~~~text
no prior observed profile:  a_t = f_t
causal_running_mean:       a_t = f_t + alpha * mean(f_s for s in H_t)
none:                      a_t = f_t
~~~

History uses unaugmented profiles, resets per response, excludes missing profiles
and uses no gold, predictions, future steps or other responses. Dimensions stay
unchanged. The running mean and alpha=0.5 are reconstruction choices. For paired
ablation set sequential_augmentation=none and hold the data, split, seed,
vocabulary, layout, normalization and training budget fixed. The synthetic
comparison validates this switch, not its scientific effectiveness.

fixtures/atr_resnet18_reconstruction.json records standard-width defaults:
base_channels=64, batch_size=32, epochs=20, AdamW learning_rate=0.001,
weight_decay=0.0001, seed=1337 and num_workers=0. The smoke uses base_channels=2,
batch_size=4, two epochs and learning_rate=0.003. Batch size one or fewer than
two train steps is rejected. Trailing singleton batches merge into the previous
batch; no train step is dropped.

Validation cross-entropy covers only observed steps with candidate labels.
Coverage and OOV/missing counts are separately reported. This restricted loss
summary is not the full DASR evaluation.

## Frozen vocabulary, checkpoint and isolation

At least two distinct train gold IDs are required. Vocabulary construction
ignores validation/test labels, stores task/tokenizer identity and a canonical
digest, and never expands the classifier during inference.

The generated checkpoint has model_state.pt and metadata.json. A primitive
envelope binds config, input shape, task/tokenizer/alignment/feature contract,
candidate vocabulary and train provenance to the state dictionary. Loading uses
torch.load(weights_only=True), verifies contract/file digests, then state keys,
shapes, dtypes and finite values. Logits, probabilities, losses, gradients and
updated state must remain finite.

Train provenance records response/case/step identities, gold IDs and full-response
profile digests. Encoding float32_le_shape_missingmask_v1 hashes little-endian
model-dtype raw values with shape, step count and missing-position markers;
negative zero is canonicalized. Integer/float JSON aliases and values that round
to identical float32 bytes cannot bypass duplicate-profile checks.

All manifest splits are checked, including non-selected inference rows. A train
response, case or step cannot be reassigned to validation/test across manifests.
Reusing a full train response profile under new IDs in another split is rejected.
Full-response duplicate profiles also cannot cross splits within one manifest.
Equal individual step vectors are permitted. The digest covers raw declared
float32 profiles, not every normalization-equivalent feature transformation.

Digests provide consistency and accidental-integrity checks. They do not
authenticate an adversary who can rewrite the checkpoint and every digest.
The loader is for this pipeline's generated checkpoints, not unknown legacy
Python-object models.

## Inference and DASR

Every selected step produces a row keyed by response_id/step_id, including
missing profiles with predicted_token_id=null. Missing rows never shift later
alignment. Metrics verify optional step_index/case/split/task identity and reject
duplicate or unknown prediction IDs.

~~~text
response DASR = correct token predictions / all gold tokens in that response
DASR macro    = mean(response DASR over responses)
~~~

OOV gold IDs and missing predictions count incorrect. Candidate coverage and
separately named token micro rate do not replace macro DASR. --no-evaluate permits
null gold and emits not_evaluated. Evaluation rejects missing gold instead of
using a labeled subset.

## Commands

Run from the repository with its initialized Python environment:

~~~bash
python -m artifacts.janus_artifact.cli atr-synthetic-smoke \
  --work-dir artifacts/test-output/atr-example-smoke --device cpu

python -m artifacts.janus_artifact.cli atr-validate \
  --manifest artifacts/test-output/atr-example-smoke/data/manifest.jsonl

python -m artifacts.janus_artifact.cli atr-freeze-vocabulary \
  --manifest artifacts/test-output/atr-example-smoke/data/manifest.jsonl \
  --output artifacts/test-output/atr-example-vocabulary.json

python -m artifacts.janus_artifact.cli atr-train \
  --manifest artifacts/test-output/atr-example-smoke/data/manifest.jsonl \
  --config artifacts/fixtures/atr_resnet18_reconstruction.json \
  --checkpoint-dir artifacts/test-output/atr-example-training --device cpu

python -m artifacts.janus_artifact.cli atr-infer \
  --manifest artifacts/test-output/atr-example-smoke/data/manifest.jsonl \
  --checkpoint-dir artifacts/test-output/atr-example-smoke/checkpoint \
  --split test --device cpu \
  --output artifacts/test-output/atr-example-predictions.jsonl \
  --metrics-output artifacts/test-output/atr-example-metrics.json
~~~

atr-split optionally creates a case-grouped split and records seed, ratios and
assignment digest. It preserves the previous recorded assignment ID as
parent_split_assignment_id. This is generated bookkeeping, not author
certification; retain existing author assignments for real evaluation.

## Remaining evidence

Real Janus evaluation still needs complete author-matched response/step/gold
records, tokenizer revision, special-token scoring, calibrated reconstruction
evidence and immutable split assignments. Full-tokenizer classification and
comparison are not implemented by this train-candidate pipeline. The 8/2/2
synthetic smoke is not the paper's dataset or an 8:1:1 experiment. Collector
integration, scientific reconstruction, exact author predictor/augmentation and
published attack rates remain unverified.

See ATR_VALIDATION.md for actual software checks.
