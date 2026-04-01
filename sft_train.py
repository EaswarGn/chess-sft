import argparse
import wandb
import torch
import re
import numpy as np
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, EvalPrediction
import chess
import sys
from typing import Dict


def main():
    parser = argparse.ArgumentParser(description="High-throughput fine-tuning for A100 80GB.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Pre-trained model identifier from Hugging Face")
    parser.add_argument("--project_name", type=str, default="chess-reasoning-v1", help="Weights & Biases project name for logging")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
    args_cli = parser.parse_args()

    if args_cli.use_wandb:
        wandb.init(
            project=args_cli.project_name, 
            name=f"ultra-sft-{args_cli.model_id.split('/')[-1]}",
            config={"model_id": args_cli.model_id}
        )

    tokenizer = AutoTokenizer.from_pretrained(args_cli.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args_cli.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )

    dataset = load_dataset("codingmonster1234/chess-reasoning-sft")

    training_args = SFTConfig(
        output_dir=f"./output-{args_cli.model_id.split('/')[-1]}",
        bf16=True,
        tf32=True,
        
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        
        eval_strategy="epoch",
        logging_steps=1,
        report_to="wandb" if args_cli.use_wandb else "none",
        per_device_eval_batch_size=1,
        eval_accumulation_steps=8,
        
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        save_strategy="epoch",
        save_total_limit=2,
        
        max_length=1024*2,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args
    )
    
    initial_metrics = trainer.evaluate()
    print("Initial Evaluation Metrics:", initial_metrics)

    trainer.train()
    
    final_metrics = trainer.evaluate()
    print("Final Evaluation Metrics:", final_metrics)
    
    if args_cli.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()