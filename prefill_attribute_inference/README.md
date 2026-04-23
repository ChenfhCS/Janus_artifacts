# Prefill-Phase Query Attribute Inference

This directory contains the implementation of the **Prefill-Phase Query Attribute Inference** attack.

The attack infers sensitive semantic attributes of user queries from reconstructed token-level sparsity patterns derived from **prefill-phase SIMA traces**.  
It does **not** require access to the victim LLM’s parameters, activations, or API outputs.

---

## Directory Organization

This directory is organized by **dataset-model** pairs:

```text
prefill_attribute_inference/
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

### Typical Contents of Each `dataset-model` Directory

Each `dataset-model` directory in `prefill_attribute_inference/` contains:

- a pretrained attribute prediction model, e.g.,
  `best_resnet18_not_pca_illness_num_classes4.pt`
- a folder containing reconstructed sparsity-pattern inputs, e.g.,
  `infer_10_val/`
- a validation CSV file, e.g.,
  `infer_10_val.csv`
- an inference script, e.g.,
  `load_model_infer_illness_10_load_npz_speed_up.py`

A typical example is:

```text
financial-llama/
├── best_resnet18_not_pca_illness_num_classes4.pt
├── infer_10_val/
├── infer_10_val.csv
└── load_model_infer_illness_10_load_npz_speed_up.py
```

---

### Running Attribute Inference

From the artifact root directory:

```bash
cd prefill_attribute_inference/<dataset-model>
python <prefill_inference_script>.py
```

Example:

```bash
cd prefill_attribute_inference/financial-llama
python load_model_infer_illness_10_load_npz_speed_up.py
```

Please use the inference script provided in the selected `dataset-model` directory.

---

### Input and Output

- The folder such as `infer_10_val/` stores the reconstructed token-level sparsity patterns used as attack inputs.
- The CSV file such as `infer_10_val.csv` stores the corresponding evaluation queries and labels.
- The pretrained `.pt` file stores the attribute predictor.
- The Python script runs inference for that setting and writes prediction results according to its built-in logic.

---

### Notes

- Each `dataset-model` directory is self-contained.
- Script names may differ slightly across different settings.
- Please run the script inside the corresponding subdirectory so that all relative paths resolve correctly.
