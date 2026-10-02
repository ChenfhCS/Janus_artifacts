# Paper-to-artifact module map

The paper describes a three-stage pipeline. The current public repository contains
selected attack-inference scripts and Git LFS pointers. The new controlled CPU
loop connects the reconstructed stages using simulated probes and explicit
oracle annotations; physical collection remains unavailable.

| Paper module | Paper role | Existing repository evidence | This milestone | Status / gap |
| --- | --- | --- | --- | --- |
| Phase distinction | Detect prefill and decoding intervals | Not present as reusable code | Kernel-duration/window proxy and clock-anchored nearest-round reference alignment | Oracle-aligned boundaries; autonomous trace-only detector and physical timeline collector absent |
| Prefill SIMA extraction | L2-based cumulative page trace | Binary payloads referenced through LFS only | Complete raw probe/mapping/calibration contract and sum of recorded eviction footprints | Relative-frequency proxy from synthetic observations; physical calibration/collection absent |
| Decoding SIMA extraction | TLB-based step-wise page trace | Binary payloads referenced through LFS only | Strict latency threshold and majority of all rounds in an oracle-aligned step | Vote grouping/minimum/ties are choices; cross-step window filter and physical calibration/collection absent |
| Prefill reconstruction | Page frequency to approximate token sparsity | One verified legacy rank tensor plus LFS pointers; no standalone implementation found | Mass-conserving representative power-law profile and declared lossy QAI rank surrogate | Exponent/distribution and adapter are reconstruction choices, without author-matched validation |
| Decoding reconstruction | Local page density to approximate token sparsity | No standalone implementation found | Same-head local density with configured ceil/prefix token placement | Radius, rounding and placement are reconstruction choices, without author-matched validation |
| Query attribute inference (QAI) | Classify attributes from normalized prefill profiles | Per-setting inference scripts and unsafe whole-object `.pt` pointers | Clean ResNet-18 training/inference/PASR path with safe state-dict checkpoints | Hyperparameters are reconstruction choices; real split manifest and complete evaluation remain absent |
| Autoregressive token recovery (ATR) | Sequential token classification | Per-setting inference scripts, CSV outputs, `.pt` pointers | Clean per-step training/inference/DASR path, explicit IDs/tokenizer/alignment, frozen train vocabulary and causal augmentation | Architecture/augmentation are reconstruction choices; real token records and full-tokenizer evaluation remain absent |
| Evaluation | PASR and per-response DASR | Selected top-10 outputs; static source proves their CSV-index-to-case-directory join | Evidence-backed one-sample manifest and strict metrics | Canonical split, non-selected predictions, and token labels remain absent; published numbers are not reproduced here |

The legacy scripts were reviewed statically and were not executed. Several scripts
load full Python objects with `torch.load(..., weights_only=False)`, rely on missing
local modules or hard-coded paths, and select high-accuracy/high-confidence examples.
Those legacy behaviors remain outside the safe pipeline.

The controlled generator fixes case-group splits before running a deterministic
integer KV toy operation. It preserves run/sample/case/split metadata for QAI,
and complete response/step/tokenizer/gold mapping for ATR and shared
mapping_provenance.json; trace records keep their own explicit phase mapping.
[UPSTREAM_RECONSTRUCTION.md](UPSTREAM_RECONSTRUCTION.md) records the paper basis
and numerical choices; [UPSTREAM_VALIDATION.md](UPSTREAM_VALIDATION.md) records
the actual CPU check. No stage presents oracle metadata as measured probes.
