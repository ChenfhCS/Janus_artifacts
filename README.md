# Artifact: Sparsity-Guided Inference Attacks

This artifact contains the implementation and pretrained models for the two inference attacks described in the paper:

1. **Prefill-Phase Query Attribute Inference (QAI)**  
2. **Decoding-Phase Autoregressive Token Recovery (ATR)**

Both attacks operate on reconstructed sparsity patterns derived from sparsity-induced memory access (SIMA) traces collected during LLM inference.  
The attacks do **not** require access to the victim LLM’s parameters, activations, or API outputs.

---

## Directory Structure

After unpacking the artifact, the directory structure is organized as follows:

```text
janus-artifact/
├── README.md
├── prefill_attribute_inference/
│   ├── README.md
│   ├── health-longchat/
│   ├── health-llama/
│   ├── health-qwen/
│   ├── financial-longchat/
│   ├── financial-llama/
│   ├── financial-qwen/
│   ├── legal-longchat/
│   ├── legal-llama/
│   └── legal-qwen/
│
└── decoding_token_recovery/
    ├── README.md
    ├── health-longchat/
    ├── health-llama/
    ├── health-qwen/
    ├── financial-longchat/
    ├── financial-llama/
    ├── financial-qwen/
    ├── legal-longchat/
    ├── legal-llama/
    └── legal-qwen/
