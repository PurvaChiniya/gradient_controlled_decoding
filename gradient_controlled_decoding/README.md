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
For controlled decoding run the model with the generated labels from detection model , using python controlled_decode.py

##Testing 
For end-to-end testing run eval_one.py
