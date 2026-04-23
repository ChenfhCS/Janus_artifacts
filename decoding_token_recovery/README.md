# Decoding-Phase Autoregressive Token Recovery

This directory contains the implementation of the **Decoding-Phase Autoregressive Token Recovery** attack.

The attack reconstructs LLM-generated responses token by token during the decoding phase using reconstructed step-wise sparsity patterns derived from **decoding-phase SIMA traces**.  
It does **not** require access to the victim LLM’s parameters, activations, or API outputs.

---

## Directory Organization

This directory is organized by **dataset-model** pairs:

```text
decoding_token_recovery/
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

Each `dataset-model` directory in `decoding_token_recovery/` contains:

- a pretrained token prediction model, e.g.,
  `best_resnet34_decoding.pt`
- an inference script, e.g.,
  `load_model_infer_filter_output_resnet18_34_50.py`
- an evaluation CSV file, e.g.,
  `top10_acc_kept.csv`
- a folder storing the decoding inputs and/or outputs, e.g.,
  `top10_cases_output/`

A typical example is:

```text
financial-llama/
├── best_resnet34_decoding.pt
├── load_model_infer_filter_output_resnet18_34_50.py
├── top10_acc_kept.csv
└── top10_cases_output/
```

---

### Running Token Recovery

From the artifact root directory:

```bash
cd decoding_token_recovery/<dataset-model>
python <decoding_inference_script>.py
```

Example:

```bash
cd decoding_token_recovery/financial-llama
python load_model_infer_filter_output_resnet18_34_50.py
```

Please use the inference script provided in the selected `dataset-model` directory.

---

### Input and Output

- The CSV file such as `top10_acc_kept.csv` stores the evaluation instances used in decoding-phase inference.
- The folder such as `top10_cases_output/` stores the corresponding case data used by the recovery pipeline.
- The pretrained `.pt` file stores the token recovery model.
- The Python script runs autoregressive token recovery for that setting and writes outputs according to its built-in logic.

---

### Notes

- Each `dataset-model` directory is self-contained.
- Script names may differ slightly across different settings.
- Please run the script inside the corresponding subdirectory so that all relative paths resolve correctly.
