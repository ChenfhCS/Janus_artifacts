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

This directory contains the implementation of the **Prefill-Phase Query Attribute Inference** attack.  
Each `dataset-model` subdirectory includes the pretrained model, the reconstructed sparsity-pattern inputs, the validation CSV file, and the inference script for that setting.

See `prefill_attribute_inference/README.md` for details.

---

#### 2. `decoding_token_recovery/`

This directory contains the implementation of the **Decoding-Phase Autoregressive Token Recovery** attack.  
Each `dataset-model` subdirectory includes the pretrained model, the reconstructed decoding inputs, the evaluation CSV file, and the inference script for that setting.

See `decoding_token_recovery/README.md` for details.

---

### Quick Start

All commands below are executed from the artifact root directory:

```bash
cd janus-artifact
```

#### Run Prefill-Phase Attribute Inference

```bash
cd prefill_attribute_inference/<dataset-model>
python <prefill_inference_script>.py
```

Example:

```bash
cd prefill_attribute_inference/financial-llama
python load_model_infer_illness_10_load_npz_speed_up.py
```

---

#### Run Decoding-Phase Token Recovery

```bash
cd decoding_token_recovery/<dataset-model>
python <decoding_inference_script>.py
```

Example:

```bash
cd decoding_token_recovery/financial-llama
python load_model_infer_filter_output_resnet18_34_50.py
```

---

### Notes

- Each `dataset-model` directory is self-contained for one evaluation setting.
- The exact script and file names may differ across settings, but the usage pattern is the same.
- Please run the inference script inside the corresponding `dataset-model` directory so that all relative paths resolve correctly.
