To reproduce baseline comparisons, first set up the environment from the repo root:
`pip install -r requirements.txt`

Login to Hugging Face before running the public dataset pipelines:
`huggingface-cli login`

## Available Baselines
- `gradSafe`: gradient-based detection plus controlled decoding, aligned to the same dataset/model/output format as the main method
- `safe_decoding`: original baseline scripts kept in their existing format

## gradSafe Detection
Run the gradSafe detection and threshold pipeline for all datasets with:
`./scripts/run_gradsafe_pipeline.sh`

To run a different model:
`./scripts/run_gradsafe_pipeline.sh meta-llama/Llama-3.2-3B-Instruct`

The script runs:
- `baselines/gradSafe/detection.py`
- `baselines/gradSafe/threshold.py`

Datasets used:
- `advbench`
- `xstest`
- `security` if `security_data.csv` exists
- `toxicchat` with `--max-samples 500` by default

You can override the toxicchat cap:
`TOXICCHAT_MAX_SAMPLES=200 ./scripts/run_gradsafe_pipeline.sh`

Outputs are saved under:
- `baselines/gradSafe/outputs/detection`
- `baselines/gradSafe/outputs/threshold`

Example outputs:
- `baselines/gradSafe/outputs/detection/advbench__meta-llama-llama-2-7b-chat-hf__scores.csv`
- `baselines/gradSafe/outputs/threshold/advbench__meta-llama-llama-2-7b-chat-hf__scores__thresholds.json`

## gradSafe Generations
Run the gradSafe generation pipeline with:
`./scripts/run_gradsafe_generation_pipeline.sh`

To run a different model:
`./scripts/run_gradsafe_generation_pipeline.sh meta-llama/Llama-3.2-3B-Instruct`

The script runs:
- `baselines/gradSafe/controlled_decode.py`

Outputs are saved under:
- `baselines/gradSafe/outputs/generations`

Example output:
- `baselines/gradSafe/outputs/generations/advbench__meta-llama-llama-2-7b-chat-hf__generations.csv`

Optional generation controls:
- `OVERWRITE=1` to replace existing generation CSVs
- `MAX_NEW_TOKENS=128` to control generation length
- `TEMPERATURE=0.7` and `TOP_P=0.9` to control sampling
- `SEED=0` to control reproducibility

Example:
`OVERWRITE=1 MAX_NEW_TOKENS=128 ./scripts/run_gradsafe_generation_pipeline.sh`

## safe_decoding
The `safe_decoding` baseline remains in its original format. Use:
- `baselines/safe_decoding/run.py`
- `baselines/safe_decoding/eval.py`
- `baselines/safe_decoding/latency.py`
