# Gradient Controlled Decoding Experiments

This repository contains:
- `gradient_controlled_decoding/`: gradient-based controlled decoding experiments.
- `baselines/safe_decoding/`: baseline generation + refusal-style evaluation.
- `sft/`: supervised fine-tuning script for Llama-style chat models.
- `results/`: example output artifacts.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Access

By default, scripts use `meta-llama/Llama-2-7b-chat-hf`.
You must have Hugging Face access configured for gated models.

## Datasets

- ToxicChat: `lmsys/toxic-chat` (`toxicchat1123`)
- XSTest: `natolambert/xstest-v2-copy`
- AdvBench: `walledai/AdvBench`

## Quick Reproduction (Baseline Pipeline)

Run baseline generation + evaluation end to end:

```bash
./scripts/reproduce.sh
```

Optional environment overrides:

```bash
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf DATASET_NAME=AdvBench ./scripts/reproduce.sh
```

Outputs:
- `<dataset>.csv`
- `<dataset>_output_with_predictions.csv`
- `<dataset>_evaluation_metrics.json`

## Manual Commands

Generate baseline outputs:

```bash
python baselines/safe_decoding/run.py \
  --data_name xstest \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --output_csv xstest.csv
```

Evaluate baseline outputs:

```bash
python baselines/safe_decoding/eval.py \
  --data_name xstest \
  --save-metrics-json
```

Run SFT training:

```bash
python sft/finetune.py \
  --dataset-path /path/to/dataset_saved_with_hf_datasets \
  --text-field text \
  --model-name meta-llama/Llama-2-7b-chat-hf \
  --output-dir ./results \
  --save-model-dir ./sft
```

Or use a Hugging Face dataset directly:

```bash
python sft/finetune.py \
  --dataset-name tatsu-lab/alpaca \
  --dataset-split train \
  --text-field text
```

## Notes

- Baseline scripts expect a `prompt` field in loaded datasets.
- `baselines/safe_decoding/eval.py` expects a `ground_truth` column in `<dataset>.csv`.
- `sft/finetune.py` is now CLI-configurable and no longer depends on external undefined preprocessors.

