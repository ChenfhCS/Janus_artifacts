# Janus QAI audit repair validation

The five static-review issues were present at source commit c8554d4 and are now
addressed in the artifact module. No legacy script or unknown checkpoint was
executed or loaded. Only synthetic data and newly generated safe checkpoints were
used. These runs validate software contracts, not the paper's scientific results.

## Changes

1. Required training provenance stores all train sample/case identities and exact
   present-payload SHA-256 values. Complete manifests are checked for cross-split
   case and identical-byte payload leakage. Independent inference manifests are
   checked against training identities and payloads, including absent records.
2. Inference requires every supplied record's attribute to match the checkpoint,
   including mixed attributes and absent-only manifests.
3. v2 safe checkpoint envelopes bind canonical metadata semantics to tensor state.
   Sorted unique labels, config/model/feature contracts, tensor shapes/dtypes and
   finite state/logits/probabilities are enforced. Checksums provide integrity,
   not authentication against a party that can rewrite the whole checkpoint.
4. The last singleton training batch is merged into its preceding batch, retaining
   every sample; configured batch size one is rejected before payload reads.
5. NPZ exact unique members and directory budgets are checked before decompression,
   bounded NPY declarations before allocation, and CRC before pickle-disabled
   NumPy loading.

## Actual validation

Environment: Python 3.12.3, PyTorch 2.8.0+cu128, NumPy 2.3.2 on the existing server.
The regression suite ran on CPU with one OpenMP/MKL thread.

From the artifacts directory:

~~~sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t .. -v
git diff --check
~~~

The final code suite reported **75 tests, 1.992 seconds, OK**:

| Suite | Passed |
| --- | ---: |
| Existing core contracts and synthetic QAI pipeline | 27 |
| Checkpoint integrity | 17 |
| NPZ budgets and NPY headers | 12 |
| All-sample training batches | 4 |
| Training provenance and attribute isolation | 15 |
| Total | 75 |

The 33-sample, batch-size-32, 32-by-32 grid regression observed a training batch of
33, a final layer4 output of (33, 16, 1, 1), train_total=33, finite gradients and
changed model parameters. The existing synthetic smoke verified gradient/update,
safe save/load, five test predictions and a PASR denominator of four present
queries. PASR values from this fixture are not scientific evidence.

The negative cases include train case/sample reassignment to test, identical
payload bytes copied under new IDs, absent training case reassignment, overlap
outside the selected inference split, mixed/absent-only attribute mismatch,
duplicate/reordered labels, metadata/state mismatch, NaN/Inf state and outputs,
duplicate ZIP members, oversized directory declarations, and a tiny member with
a huge declared NPY shape. Oversized archive tests use small headers or mocks,
not actual ZIP bombs. The diff whitespace check passed.

## Limits

The payload contract compares exact NPZ bytes; re-encoded copies of equal arrays
are not identified by that digest. Identity checks depend on stable original IDs.
Old v1 checkpoints lack required bound provenance and must be retrained. The
largest training batch can be configured batch_size + 1. No real query CSV,
trace corpus, legacy weights, large model, or third-party target was used.
