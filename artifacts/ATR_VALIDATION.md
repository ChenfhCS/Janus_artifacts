# ATR software validation

Date: 2026-10-01. Checks ran on the initialized AutoDL Python 3.12.3,
PyTorch 2.8.0+cu128 and NumPy 2.3.2 environment, on CPU with
OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. No legacy attack script, unknown model,
large download or third-party attack was run.

## Full regression suite

The complete output remains at test-output/atr-grid32-final-regression.log.
From artifacts, with the repository parent as the unittest top-level directory:

~~~bash
python -m unittest discover -s tests -t .. -v
~~~

Result: 171 tests passed in 7.131 seconds (exit 0), including the earlier 75
QAI/offline/NPZ checks. ATR coverage includes train-only vocabulary/width,
response/case/step isolation, cross-manifest train provenance, float32 digest
aliases, strict tokenizer/alignment, absent validation, safe finite checkpoints
and outputs, causal augmentation, internal missing-step alignment, and strict
JSON type identity across checkpoint contracts and training source evidence.

The 33-train-step/batch-32 ATR checks cover both a 4 x 4 grid (input
[4, 4, 4]) and a separate 32 x 32 grid (input [4, 32, 32]). The latter tiles
generated synthetic profiles and updates the declared layer/head IDs; it is not
a recorded 32-layer side-channel trace. Its test asserts input shape, 33 train
steps, two epochs with train_total=33 in each, finite gradients and parameter
updates. Batch size one is rejected. The QAI singleton-batch and bounded NPZ
regressions continue to pass.

An independent CPU run used that same test helper, base_channels=2, batch_size=32,
epochs=2, learning_rate=0.001 and seed=19. The persisted evidence is
test-output/atr-train33-grid32/report.json:

| Item | Actual 32 x 32 regression result |
| --- | --- |
| Synthetic train responses / steps | 11 / 33 |
| Model input shape | [4, 32, 32] |
| Per-epoch train totals | [33, 33] |
| Finite gradient / parameter update | both true |
| Safe checkpoint reload | passed with weights_only=True |
| Training runtime on CPU | 0.08780859410762787 seconds |
| Data-check/training/reload duration | 2.759230762720108 seconds |

This train-only regression verifies batching and finite learning behavior.
It does not evaluate token-recovery accuracy or reproduce the paper.

## Actual CLI smoke

From the repository:

~~~bash
python -m artifacts.janus_artifact.cli atr-synthetic-smoke \
  --work-dir artifacts/test-output/atr-final-smoke --device cpu
~~~

Exit 0: finite gradient observed, parameters changed, and generated checkpoint
reloaded for inference. Both final checkpoints were also reloaded using the final
type-preserving source; all six predictions and the entire metric reports matched. Report: test-output/atr-final-smoke/report.json.
Generated manifest, weights and predictions stay in that output directory;
they are not source-package material.

| Item | Actual result |
| --- | --- |
| Synthetic responses | train 8, validation 2, test 2 |
| Training budget | two epochs, all 24 train steps each epoch |
| Model | ResNet18 reconstruction, base_channels=2, 11,622 parameters |
| Input shape | 4 token channels x 4 layers x 4 KV heads |
| Training runtime | 0.20182887464761734 seconds on CPU |
| Frozen candidates | token IDs 11 and 17 |
| Test rows / full gold denominator | 6 / 6 |
| Candidate coverage | 5/6 |
| OOV gold counted wrong | 1 |
| Missing-profile prediction counted wrong | 1 |
| DASR macro / separately named micro | 1/6 / 1/6 |

The tiny classifier result proves learning/save/load/inference and denominator
policies run. It provides no attack-accuracy conclusion.

## Paired augmentation switch

The no-augmentation check used the same final manifest, split, vocabulary,
shape, seed=19, learning_rate=0.003, base_channels=2, batch_size=4 and two epochs.
Only sequential_augmentation changed from causal_running_mean to none.

Report: test-output/atr-final-noaug/report.json. CPU training took 0.1969 seconds;
training/reload/inference together took 1.0954 seconds. Finite gradients, parameter
updates and two safe reloads passed. Both modes produced macro DASR 1/6 and retained
six gold tokens, one OOV and one missing prediction. Validation was 3/6.
This validates the implementation switch and supports no augmentation-benefit
claim.

## Limits

All runtime inputs were generated fixtures. No author test responses, tokenizer
alignment, real token labels, collector or scientific reconstruction pipeline
were available. ATR_RECONSTRUCTION.md records the paper basis and explicit
reconstruction choices.
