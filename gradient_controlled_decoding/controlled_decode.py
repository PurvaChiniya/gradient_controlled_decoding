import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SUPPORTED_DATASETS = ["advbench", "toxicchat", "xstest", "security"]
DEFAULT_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
DEFAULT_DETECTION_DIR = Path("outputs/detection")
DEFAULT_EVAL_DIR = Path("outputs/eval")
DEFAULT_OUTPUT_DIR = Path("outputs/generations")
POSITIVE_FIRST_TOKENS = ["Sure,", "Absolutely,", "Certainly,"]
NEGATIVE_FIRST_TOKENS = ["Sorry,", "Unfortunately,", "I cannot,"]
PROMPT_COLUMNS = {
    "advbench": "prompt",
    "toxicchat": "user_input",
    "xstest": "prompt",
    "security": "prompt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run controlled decoding from saved detection scores and thresholds."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model ID to load. Defaults to {DEFAULT_MODEL_ID}.",
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default=None,
        help="Generate for a single dataset. If omitted, all matching datasets for the model are used.",
    )
    parser.add_argument(
        "--scores-csv",
        default=None,
        help="Explicit detection scores CSV to use for a single run.",
    )
    parser.add_argument(
        "--thresholds-json",
        default=None,
        help="Explicit eval JSON containing joint sure/sorry thresholds for a single run.",
    )
    parser.add_argument(
        "--detection-dir",
        default=str(DEFAULT_DETECTION_DIR),
        help=f"Directory containing detection score CSVs. Defaults to {DEFAULT_DETECTION_DIR}.",
    )
    parser.add_argument(
        "--eval-dir",
        default=str(DEFAULT_EVAL_DIR),
        help=f"Directory containing eval threshold JSONs. Defaults to {DEFAULT_EVAL_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where generation CSVs will be written. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for first-token selection and generation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on rows generated from each scores CSV.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt. By default no system prompt is added.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing generations CSV if present.",
    )
    return parser.parse_args()


def slugify(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower()


def load_model(model_id):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def get_input_device(model):
    return next(model.parameters()).device


def build_generation_prompt(tokenizer, prompt, first_token, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return f"{prompt_text}{first_token}"

    prompt_prefix = ""
    if system_prompt:
        prompt_prefix = f"SYSTEM: {system_prompt}\n"
    return f"{prompt_prefix}USER: {prompt}\nASSISTANT: {first_token}"


def infer_dataset_from_scores(scores_csv):
    stem = Path(scores_csv).stem
    dataset = stem.split("__", 1)[0]
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unable to infer dataset from scores CSV name: {scores_csv}")
    return dataset


def resolve_eval_json(dataset, model_slug, eval_dir):
    patterns = [
        f"{dataset}__{model_slug}__scores__eval_metrics.json",
        f"{dataset}__{dataset}__{model_slug}__scores__eval_metrics.json",
        f"{dataset}__{model_slug}__eval_metrics.json",
        f"{dataset}__{dataset}__{model_slug}__eval_metrics.json",
    ]
    for pattern in patterns:
        candidate = eval_dir / pattern
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find eval thresholds JSON for dataset={dataset}, model={model_slug} in {eval_dir}"
    )


def resolve_scores_csv(dataset, model_slug, detection_dir):
    candidate = detection_dir / f"{dataset}__{model_slug}__scores.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find detection scores CSV for dataset={dataset}, model={model_slug} in {detection_dir}"
        )
    return candidate


def build_run_specs(args):
    detection_dir = Path(args.detection_dir)
    eval_dir = Path(args.eval_dir)
    model_slug = slugify(args.model_id)

    if args.scores_csv:
        dataset = args.dataset or infer_dataset_from_scores(args.scores_csv)
        thresholds_json = (
            Path(args.thresholds_json)
            if args.thresholds_json
            else resolve_eval_json(dataset, model_slug, eval_dir)
        )
        return [
            {
                "dataset": dataset,
                "scores_csv": Path(args.scores_csv),
                "thresholds_json": thresholds_json,
            }
        ]

    datasets = [args.dataset] if args.dataset else SUPPORTED_DATASETS
    run_specs = []
    for dataset in datasets:
        try:
            scores_csv = resolve_scores_csv(dataset, model_slug, detection_dir)
            thresholds_json = resolve_eval_json(dataset, model_slug, eval_dir)
        except FileNotFoundError:
            continue
        run_specs.append(
            {
                "dataset": dataset,
                "scores_csv": scores_csv,
                "thresholds_json": thresholds_json,
            }
        )

    if not run_specs:
        raise FileNotFoundError(
            f"No matching detection/eval files found for model={args.model_id}."
        )
    return run_specs


def load_thresholds(thresholds_json):
    with Path(thresholds_json).open("r", encoding="utf-8") as f:
        data = json.load(f)
    best_thresholds = data.get("best_thresholds", {})
    if "sure" not in best_thresholds or "sorry" not in best_thresholds:
        raise ValueError(f"Missing best thresholds in {thresholds_json}")
    return float(best_thresholds["sure"]), float(best_thresholds["sorry"])


def ensure_prompt_column(df, dataset):
    prompt_column = PROMPT_COLUMNS[dataset]
    if prompt_column not in df.columns:
        raise ValueError(
            f"Expected prompt column '{prompt_column}' in scores CSV for dataset={dataset}"
        )
    if prompt_column != "prompt":
        df = df.rename(columns={prompt_column: "prompt"})
    return df


def pick_label(row, sure_threshold, sorry_threshold):
    return int((float(row["sure"]) >= sure_threshold) and (float(row["sorry"]) >= sorry_threshold))


def generate_one(
    model,
    tokenizer,
    prompt,
    label,
    max_new_tokens,
    temperature,
    top_p,
    system_prompt=None,
):
    first_token = random.choice(POSITIVE_FIRST_TOKENS if label == 0 else NEGATIVE_FIRST_TOKENS)
    prompt_text = build_generation_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        first_token=first_token,
        system_prompt=system_prompt,
    )
    input_device = get_input_device(model)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(input_device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    assistant_suffix = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    assistant_response = f"{first_token} {assistant_suffix}".strip()
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return first_token, assistant_response, full_text


def run_generation_for_spec(args, spec, model, tokenizer):
    scores_csv = spec["scores_csv"]
    thresholds_json = spec["thresholds_json"]
    dataset = spec["dataset"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sure_threshold, sorry_threshold = load_thresholds(thresholds_json)
    df = pd.read_csv(scores_csv)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()
    else:
        df = df.copy()
    df = ensure_prompt_column(df, dataset)

    required_columns = {"prompt", "sure", "sorry"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {scores_csv}: {missing_columns}")

    df["generation_label"] = df.apply(
        lambda row: pick_label(row, sure_threshold, sorry_threshold),
        axis=1,
    )
    df["generation_sure_threshold"] = sure_threshold
    df["generation_sorry_threshold"] = sorry_threshold

    seed_tokens = []
    assistant_responses = []
    decoded_outputs = []

    input_device = get_input_device(model)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Generating {dataset}"):
        first_token, assistant_response, full_text = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=row.prompt,
            label=row.generation_label,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            system_prompt=args.system_prompt,
        )
        seed_tokens.append(first_token)
        assistant_responses.append(assistant_response)
        decoded_outputs.append(full_text)

    df["seed_first_token"] = seed_tokens
    df["generated_response"] = assistant_responses
    df["decoded_output"] = decoded_outputs

    model_slug = slugify(args.model_id)
    output_path = output_dir / f"{dataset}__{model_slug}__generations.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Re-run with --overwrite to replace it."
        )
    df.to_csv(output_path, index=False)
    return output_path


def main():
    args = parse_args()
    run_specs = build_run_specs(args)
    model, tokenizer = load_model(args.model_id)

    for spec in run_specs:
        output_path = run_generation_for_spec(args, spec, model, tokenizer)
        print(f"Saved generations to {output_path}")


if __name__ == "__main__":
    main()
