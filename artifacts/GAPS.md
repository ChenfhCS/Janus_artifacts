# Known gaps and non-claims

This milestone provides a testable offline contract, not a reproduction of the
Janus side channel or the paper's reported attack rates.

## Evidence gaps

- The public checkout stores `.npz` traces and `.pt` models as Git LFS pointers.
  One 1,060,689-byte prefill rank tensor was safely materialized and inspected;
  other payloads and every `.pt` remain unfetched and unexecuted.
- No reusable phase detector, probe calibration, L2/TLB collector, or token-level
  reconstruction implementation was found in the public repository snapshot.
  The new controlled loop implements reference alignment, page-event operators
  and representative token profiles from simulated observations. It supplies
  explicit toy mapping/calibration constants, without a physical collector or
  author-matched reconstruction parameters.

- Upstream training code, split manifests, candidate-vocabulary construction,
  dependency pins, and complete evaluation inputs are absent. The new QAI and ATR
  trainers are explicit reconstructions and do not replace those missing records.
- Existing CSVs are selected outputs, including "top10" cases. They cannot support
  unbiased reproduction of the paper's aggregate PASR/DASR.
- Static source inspection proves that the selected output CSV `index` maps to the
  `case_{index}` NPZ directory, including case 57. The new dataset manifest records
  that join with source and excerpt hashes. No versioned split manifest proves which
  canonical evaluation split should contain it.

## Deliberate non-claims

- Synthetic fixtures exercise contracts, numerical operators and small CPU
  training/inference loops. Their metrics describe toy software behavior.
- `replay` validates and canonicalizes already-recorded arrays; it does not execute
  a GPU spy, reconstruct token-level sparsity, or reproduce a real side channel. The
  QAI and ATR smokes train only on generated fixtures. The separate controlled
  upstream module reconstructs representative profiles under declared choices,
  using oracle-aligned phase boundaries; it does not add a physical spy or an
  autonomous trace-only phase detector.
- A random or prior-based accuracy is not a privacy guarantee. No privacy threshold,
  defense claim, or scientific conclusion is established here.
- The framework never loads legacy `.pt`, pickle, or unknown NumPy payloads. Its own
  state dictionaries and primitive contract envelopes are hash-bound and loaded
  with `weights_only=True`.

## Next evidence needed

1. A versioned, non-selected split manifest keyed by stable original sample and
   response-level group identifiers.
2. Physical collection/calibration evidence, trace-only segmentation parameters
   and author-matched reconstruction parameters and validation. Current power-law
   and local-density operators are explicit reconstruction choices.
3. Monitored layer/head ordering and token-width/padding policy for QAI features.
4. Author-matched train ATR token records, tokenizer revision, step/token alignment,
   special-token scoring, reconstruction evidence and immutable response splits.
   The train-candidate pipeline does not implement the full-tokenizer comparison.
5. Full held-out predictions needed to reproduce PASR/DASR with declared denominators.

Checkpoint/profile digests check consistency and accidental integrity. They do
not authenticate an adversary who can rewrite the metadata, state and digests.
