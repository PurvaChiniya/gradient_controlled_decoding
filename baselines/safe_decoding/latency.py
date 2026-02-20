import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(data_name: str):
    if data_name == "AdvBench":
        dataset = load_dataset("walledai/AdvBench", split="train")
    elif data_name == "security":
        dataset = load_dataset("json", data_files="data_security.json")["train"]
    elif data_name == "toxicchat":
        dataset = load_dataset("lmsys/toxic-chat", "toxicchat1123", split="test")
    elif data_name == "xstest":
        dataset = load_dataset("natolambert/xstest-v2-copy", split="prompts")
    else:
        raise ValueError("Unsupported dataset name. Choose from 'AdvBench', 'security', 'toxicchat', or 'xstest'.")
    return dataset["prompt"]


def build_input(prompt: str) -> str:
    return (
        "You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. "
        "You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. "
        "If you think there might not be a correct answer, you say so. Since you are autoregressive, each token "
        "you produce is another opportunity to use computation, therefore you always spend a few sentences "
        "explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a question."
        f"\n### Input: ```{prompt}```\n### Output:"
    )


def maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, p):
    series = pd.Series(values)
    return float(series.quantile(p))


def main():
    parser = argparse.ArgumentParser(description="Measure generation latency per decoding method.")
    parser.add_argument("--data_name", type=str, required=True, help="Dataset name: AdvBench, security, toxicchat, xstest")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="HF model name or local path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device, e.g. cuda:0 or cpu")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max generated tokens per prompt")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of prompts to evaluate")
    parser.add_argument("--warmup_samples", type=int, default=5, help="Warmup prompts (not included in final stats)")
    parser.add_argument("--output_json", type=str, default="latency_summary.json", help="Output JSON for aggregate latency metrics")
    parser.add_argument("--output_csv", type=str, default="latency_per_prompt.csv", help="Output CSV for per-prompt latency records")
    args = parser.parse_args()

    device = torch.device(args.device)
    prompts = load_prompts(args.data_name)
    prompts = prompts[: min(len(prompts), args.num_samples + args.warmup_samples)]
    if not prompts:
        raise ValueError("No prompts found for latency evaluation.")

    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=model_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    methods = {
        "greedy": dict(max_new_tokens=args.max_new_tokens, do_sample=False),
        "top_p": dict(temperature=0.7, top_p=0.9, max_new_tokens=args.max_new_tokens, repetition_penalty=1.3, do_sample=True),
        "top_k": dict(temperature=0.7, top_k=10, max_new_tokens=args.max_new_tokens, repetition_penalty=1.3, do_sample=True),
    }

    per_prompt_rows = []
    aggregated = {name: {"latency_s": [], "generated_tokens": []} for name in methods}

    for idx, prompt in enumerate(tqdm(prompts, desc="Latency run")):
        prompt_text = build_input(prompt)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = int(inputs["input_ids"].shape[-1])

        for method_name, kwargs in methods.items():
            maybe_sync(device)
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**inputs, **kwargs)
            maybe_sync(device)
            elapsed = time.perf_counter() - start

            generated_len = int(outputs.shape[-1]) - input_len
            if generated_len < 0:
                generated_len = 0

            is_warmup = idx < args.warmup_samples
            per_prompt_rows.append(
                {
                    "prompt_index": idx,
                    "is_warmup": is_warmup,
                    "method": method_name,
                    "latency_s": elapsed,
                    "generated_tokens": generated_len,
                }
            )
            if not is_warmup:
                aggregated[method_name]["latency_s"].append(elapsed)
                aggregated[method_name]["generated_tokens"].append(generated_len)

    summary = {}
    for method_name, stats in aggregated.items():
        latencies = stats["latency_s"]
        tokens = stats["generated_tokens"]
        if not latencies:
            continue
        total_time = sum(latencies)
        total_tokens = sum(tokens)
        summary[method_name] = {
            "samples": len(latencies),
            "avg_latency_s": float(sum(latencies) / len(latencies)),
            "p50_latency_s": percentile(latencies, 0.50),
            "p95_latency_s": percentile(latencies, 0.95),
            "avg_generated_tokens": float(total_tokens / len(tokens)) if tokens else 0.0,
            "tokens_per_second": float(total_tokens / total_time) if total_time > 0 else 0.0,
        }

    output_json = Path(args.output_json)
    output_json.write_text(json.dumps(summary, indent=2))

    output_csv = Path(args.output_csv)
    pd.DataFrame(per_prompt_rows).to_csv(output_csv, index=False)

    print("Latency summary (non-warmup prompts):")
    print(pd.DataFrame(summary).T.to_string())
    print(f"\nSaved summary JSON: {output_json}")
    print(f"Saved per-prompt CSV: {output_csv}")


if __name__ == "__main__":
    main()
