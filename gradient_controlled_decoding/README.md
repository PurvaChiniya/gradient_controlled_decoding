To reproduce the results, setup the environment using 
pip install -r requirements.txt

Login to Hugging Face before running the public dataset pipelines:
`huggingface-cli login`

`advbench`, `toxicchat`, and `xstest` are loaded directly from Hugging Face:
- `walledai/AdvBench`
- `lmsys/toxic-chat` (`toxicchat1123`)
- `natolambert/xstest-v2-copy`

`security` still uses the internal local file `security_data.csv`.

##Detection Model
Run the detection model with:
`python detection.py --dataset advbench`

Optional arguments:
- `--model-id` to override the default model `meta-llama/Llama-2-7b-chat-hf`
- `--output-dir` to choose where run artifacts are saved
- `--max-samples` to cap the number of dataset rows scored

The script saves one CSV and one metrics JSON per dataset/model configuration, for example:
- `outputs/detection/advbench__meta-llama-llama-2-7b-chat-hf__scores.csv`
- `outputs/detection/advbench__meta-llama-llama-2-7b-chat-hf__metrics.json`

Example for running `toxicchat` on only 500 samples:
`python detection.py --dataset toxicchat --max-samples 500`

The metrics JSON also includes scoring latency for the dataset. This latency excludes:
- model loading
- dataset loading
- critical-parameter computation in `find_critical_para`

Latency is reported for each response branch (`sure`, `sorry`) and for combined scoring only.

##Threshold 
Run threshold analysis on a detection output CSV with:
`python threshold.py --dataset advbench --scores-csv outputs/detection/advbench__meta-llama-llama-2-7b-chat-hf__scores.csv`

Optional arguments:
- `--score-columns sure sorry` to choose which score columns to analyze
- `--output-dir` to choose where threshold artifacts are saved

The threshold script:
- rebuilds ground-truth labels based on the selected dataset
- computes the best threshold by max F1 for each score column
- saves a threshold sweep plot and a precision-recall plot for each score column
- saves a JSON summary with thresholds and AUPRC values

Example outputs:
- `outputs/threshold/advbench__advbench__meta-llama-llama-2-7b-chat-hf__thresholds.json`
- `outputs/threshold/advbench__advbench__meta-llama-llama-2-7b-chat-hf__sure__threshold_sweep.png`
- `outputs/threshold/advbench__advbench__meta-llama-llama-2-7b-chat-hf__sure__pr_curve.png`

##Controlled Decoding
Run controlled decoding from the saved detection scores and joint thresholds with:
`python controlled_decode.py --model-id meta-llama/Llama-2-7b-chat-hf --dataset advbench`

Optional arguments:
- `--output-dir` to choose where generation CSVs are saved
- `--max-new-tokens` to control generation length
- `--temperature` and `--top-p` to control sampling
- `--max-samples` to cap the number of rows generated
- `--system-prompt` to add a system prompt. By default no system prompt is added.

The controlled decoding script:
- reads the saved detection CSV from `outputs/detection`
- reads the best joint `sure` and `sorry` thresholds from `outputs/eval`
- rebuilds the binary generation label from those thresholds
- generates one response per prompt
- saves one generations CSV per dataset/model configuration

Example output:
- `outputs/generations/advbench__meta-llama-llama-2-7b-chat-hf__generations.csv`

To run generation for all datasets for one model, use:
`./scripts/run_generation_pipeline.sh`

To run generation for a different model, pass the model ID:
`./scripts/run_generation_pipeline.sh meta-llama/Llama-3.2-3B-Instruct`

The generation pipeline uses:
- `advbench`
- `xstest`
- `security` if `security_data.csv` exists
- `toxicchat` with `--max-samples 500` by default

You can override the toxicchat cap when using the script:
`TOXICCHAT_MAX_SAMPLES=200 ./scripts/run_generation_pipeline.sh`

##Testing 
Run joint threshold evaluation on a detection output CSV with:
`python eval.py --dataset advbench --scores-csv outputs/detection/advbench__meta-llama-llama-2-7b-chat-hf__scores.csv`

Optional arguments:
- `--num-thresholds` to control the size of the joint threshold sweep
- `--output-dir` to choose where evaluation summaries are saved

The evaluation script:
- rebuilds ground-truth labels based on the selected dataset
- searches over joint `sure` and `sorry` thresholds
- reports the best threshold pair by max F1
- saves a JSON summary with F1, precision, recall, false positives, attack success rate, and confusion-matrix percentages over the full dataset

To evaluate the results reported from the paper pipeline, run this `eval.py` script on the detection output CSV for the dataset/model configuration you want to report.
