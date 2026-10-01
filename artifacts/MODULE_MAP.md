# Paper-to-artifact module map

The paper describes a three-stage pipeline. The current public repository contains
selected attack-inference scripts and Git LFS pointers, not an end-to-end system.

| Paper module | Paper role | Existing repository evidence | This milestone | Status / gap |
| --- | --- | --- | --- | --- |
| Phase distinction | Detect prefill and decoding intervals | Not present as reusable code | Phase field and shape validation | Collection/timestamp detector absent |
| Prefill SIMA extraction | L2-based cumulative page trace | Binary payloads referenced through LFS only | Provenance and cumulative-trace contract | Probe calibration and collection absent |
| Decoding SIMA extraction | TLB-based step-wise page trace | Binary payloads referenced through LFS only | Provenance and step-wise-trace contract | Probe calibration and collection absent |
| Prefill reconstruction | Page frequency to approximate token sparsity | One verified legacy rank tensor plus LFS pointers; no standalone implementation found | Safe external-NPZ adapter and input/output boundary | Scientific algorithm and parameters absent |
| Decoding reconstruction | Local page density to approximate token sparsity | No standalone implementation found | Input/output boundary documented | Scientific algorithm and parameters absent |
| Query attribute inference (QAI) | Classify attributes from normalized prefill profiles | Per-setting inference scripts and unsafe whole-object `.pt` pointers | Clean ResNet-18 training/inference/PASR path with safe state-dict checkpoints | Hyperparameters are reconstruction choices; real split manifest and complete evaluation remain absent |
| Autoregressive token recovery (ATR) | Sequential token classification | Per-setting inference scripts, CSV outputs, `.pt` pointers | Clean per-step training/inference/DASR path, explicit IDs/tokenizer/alignment, frozen train vocabulary and causal augmentation | Architecture/augmentation are reconstruction choices; real token records and full-tokenizer evaluation remain absent |
| Evaluation | PASR and per-response DASR | Selected top-10 outputs; static source proves their CSV-index-to-case-directory join | Evidence-backed one-sample manifest and strict metrics | Canonical split, non-selected predictions, and token labels remain absent; published numbers are not reproduced here |

The legacy scripts were reviewed statically and were not executed. Several scripts
load full Python objects with `torch.load(..., weights_only=False)`, rely on missing
local modules or hard-coded paths, and select high-accuracy/high-confidence examples.
Those legacy behaviors remain outside the safe pipeline.
