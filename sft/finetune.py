import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


@dataclass
class FineTuneConfig:
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_split: str = "train"
    text_field: str = "text"
    output_dir: str = "./results"
    save_model_dir: str = "./sft"
    new_model_name: str = "llama-2-7b-sft-alexa"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    logging_steps: int = 25
    save_steps: int = 100
    max_steps: int = -1
    gradient_checkpointing: bool = True
    group_by_length: bool = True
    lr_scheduler_type: str = "cosine"
    packing: bool = False
    max_seq_length: Optional[int] = None
    report_to: str = "none"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    device_map: str = "auto"
    fp16: bool = False
    bf16: bool = False


def parse_args() -> FineTuneConfig:
    parser = argparse.ArgumentParser(description="QLoRA SFT training entrypoint.")
    parser.add_argument("--model-name", default=FineTuneConfig.model_name)
    parser.add_argument("--dataset-path", default=None, help="Path to a dataset saved with datasets.save_to_disk.")
    parser.add_argument("--dataset-name", default=None, help="HF dataset name, used if --dataset-path is not provided.")
    parser.add_argument("--dataset-split", default=FineTuneConfig.dataset_split)
    parser.add_argument("--text-field", default=FineTuneConfig.text_field)
    parser.add_argument("--output-dir", default=FineTuneConfig.output_dir)
    parser.add_argument("--save-model-dir", default=FineTuneConfig.save_model_dir)
    parser.add_argument("--new-model-name", default=FineTuneConfig.new_model_name)
    parser.add_argument("--num-train-epochs", type=int, default=FineTuneConfig.num_train_epochs)
    parser.add_argument("--per-device-train-batch-size", type=int, default=FineTuneConfig.per_device_train_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=FineTuneConfig.gradient_accumulation_steps)
    parser.add_argument("--learning-rate", type=float, default=FineTuneConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=FineTuneConfig.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=FineTuneConfig.max_grad_norm)
    parser.add_argument("--warmup-ratio", type=float, default=FineTuneConfig.warmup_ratio)
    parser.add_argument("--logging-steps", type=int, default=FineTuneConfig.logging_steps)
    parser.add_argument("--save-steps", type=int, default=FineTuneConfig.save_steps)
    parser.add_argument("--max-steps", type=int, default=FineTuneConfig.max_steps)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--report-to", default=FineTuneConfig.report_to)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    return FineTuneConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        text_field=args.text_field,
        output_dir=args.output_dir,
        save_model_dir=args.save_model_dir,
        new_model_name=args.new_model_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
        report_to=args.report_to,
        use_4bit=not args.no_4bit,
        fp16=args.fp16,
        bf16=args.bf16,
    )


def load_training_dataset(config: FineTuneConfig):
    if config.dataset_path:
        return load_from_disk(config.dataset_path)
    if config.dataset_name:
        return load_dataset(config.dataset_name, split=config.dataset_split)
    raise ValueError("Provide either --dataset-path or --dataset-name.")


def build_model_and_tokenizer(config: FineTuneConfig):
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    quantization_config = None
    if config.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map=config.device_map,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def main():
    config = parse_args()
    dataset = load_training_dataset(config)
    if config.text_field not in dataset.column_names:
        raise ValueError(f"Column '{config.text_field}' not found. Available: {dataset.column_names}")

    model, tokenizer = build_model_and_tokenizer(config)
    peft_config = LoraConfig(
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        r=config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_arguments = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        group_by_length=config.group_by_length,
        lr_scheduler_type=config.lr_scheduler_type,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        gradient_checkpointing=config.gradient_checkpointing,
        optim="paged_adamw_32bit",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field=config.text_field,
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config.packing,
    )
    trainer.train()
    trainer.model.save_pretrained(config.new_model_name)
    trainer.save_model(config.save_model_dir)


if __name__ == "__main__":
    main()
