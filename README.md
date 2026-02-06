# Artifact: Sparsity-Guided Inference Attacks

This artifact contains the implementation and pretrained models for the two inference attacks described in the paper:

1. **Prefill-Phase Query Attribute Inference (QAI)**  
2. **Decoding-Phase Autoregressive Token Recovery (ATR)**

Both attacks operate on *reconstructed sparsity patterns* derived from sparsity-induced memory access (SIMA) traces collected during LLM inference.  
The attacks do **not** require access to the victim LLM’s parameters, activations, or API outputs.

---

## Directory Structure

After unpacking the artifact, the directory structure should be exactly as follows:

```
pea-artifact/
├── README.md
│
├── prefill_attribute_inference/
│ ├── infer_20_val/
│ ├── attribute_predictor.pt
│ ├── load_model_infer_illness.py
│ ├── health_short_4_illness_200_val.csv
│ ├── health_short_4_illness_200_val_with_pred.csv
│ └── readMe.md
│
└── decoding_token_recovery/
├── infer_20_val/
├── all_data/
├── token_predictor.pt
├── load_model_infer.py
├── build_label_mapping.py
├── health_queries_val_20.csv
├── health_queries_val_20_with_pred.csv
└── readMe.md
```

All commands described below are executed from the `pea-artifact/` root directory.

---

## 1. Prefill-Phase Prompt Attribute Inference

**Directory:** `prefill_attribute_inference/`

This component implements the *Prompt Attribute Inference (PAI)* attack, which infers sensitive semantic attributes of user queries using sparsity patterns reconstructed from prefill-phase SIMA traces.

### Files

- **`infer_20_val/`**  
  Contains reconstructed token-level sparsity patterns for 20 validation queries.  
  These sparsity patterns are recovered from SIMA traces and serve as the input to the attribute prediction model.

- **`attribute_predictor.pt`**  
  A pretrained MLP classifier that maps reconstructed sparsity patterns to query attributes.  
  In this artifact, the inferred attribute is **illness category**.

- **`load_model_infer_illness.py`**  
  Script that loads the pretrained attribute predictor and performs inference.

- **`health_short_4_illness_200_val.csv`**  
  Original validation queries with ground-truth labels.

- **`health_short_4_illness_200_val_with_pred.csv`**  
  Attribute inference results produced by the attack.

### Running Attribute Inference

```bash
cd prefill_attribute_inference
python load_model_infer_illness.py
```
---

## 2. Decoding-Phase Autoregressive Token Recovery

**Directory:** `decoding_token_recovery/`

This component implements the *Autoregressive Token Recovery (ATR)* attack, which reconstructs LLM-generated responses token by token during the decoding phase using sparsity patterns reconstructed from decoding-phase SIMA traces.

### Files

- **`infer_20_val/`**  
  Contains reconstructed *step-wise sparsity patterns* for the decoding phase.  
  The data correspond to 20 validation queries, each with 300 decoding steps.

- **`token_predictor.pt`**  
  A pretrained token-level classifier used for autoregressive token recovery.  
  At each decoding step, the model performs **1758-way classification**.

- **`load_model_infer.py`**  
  Script that loads the pretrained token prediction model and performs autoregressive token recovery.

- **`build_label_mapping.py`**  
  Utility script used to construct the mapping between token IDs and classifier labels.

- **`health_queries_val_20.csv`**  
  Original validation queries used for decoding-phase evaluation.

- **`health_queries_val_20_with_pred.csv`**  
  Output file containing reconstructed response tokens produced by the attack.

- **`all_data/`**  
  Auxiliary data used during inference.

### Attack Configuration

- Number of validation queries: **20**
- Generated tokens per query: **300**
- Vocabulary size: **1758 tokens**
- Prediction granularity: **one token per decoding step**

Each decoding step is treated as an independent 1758-class classification task.

### Running Token Recovery

```bash
cd decoding_token_recovery
python load_model_infer.py
```
