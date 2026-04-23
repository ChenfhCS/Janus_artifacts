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
```

---

### Components

#### 1. `prefill_attribute_inference/`

This directory contains the implementation of the **Prefill-Phase Prompt Attribute Inference** attack.
It infers sensitive semantic attributes of user queries from reconstructed token-level sparsity patterns derived from prefill-phase SIMA traces.

See `prefill_attribute_inference/README.md` for detailed instructions.

---

#### 2. `decoding_token_recovery/`

This directory contains the implementation of the **Decoding-Phase Autoregressive Token Recovery** attack.
It reconstructs generated response tokens step by step from reconstructed decoding-phase sparsity patterns derived from SIMA traces.

See `decoding_token_recovery/README.md` for detailed instructions.

---

### Quick Start

All commands below are executed from the artifact root directory:

```bash
cd janus-artifact
```

#### Run Prefill-Phase Attribute Inference

For a specific dataset-model pair:

```bash
cd prefill_attribute_inference/<dataset-model>
python load_model_infer_illness.py
```

Example:

```bash
cd prefill_attribute_inference/health-qwen
python load_model_infer_illness.py
```

---

#### Run Decoding-Phase Token Recovery

For a specific dataset-model pair:

```bash
cd decoding_token_recovery/<dataset-model>
python load_model_infer.py
```

Example:

```bash
cd decoding_token_recovery/health-qwen
python load_model_infer.py
```

---

### Notes

- Each `dataset-model` directory is self-contained for one evaluation setting.
- The prefill-phase and decoding-phase attacks are separated into different top-level directories for clarity.
- Output prediction files are generated within the corresponding `dataset-model` directory.
- Please run the scripts inside the corresponding `dataset-model` directory so that all relative paths resolve correctly.
