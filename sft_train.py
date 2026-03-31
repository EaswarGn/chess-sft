import argparse
import wandb
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="High-throughput fine-tuning for A100 80GB.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--project_name", type=str, default="chess-reasoning-v1")
    args_cli = parser.parse_args()

    wandb.init(
        project=args_cli.project_name, 
        name=f"ultra-sft-{args_cli.model_id.split('/')[-1]}",
        config={"model_id": args_cli.model_id}
    )

    # 1. Load Tokenizer & Model with A100-specific optimizations
    tokenizer = AutoTokenizer.from_pretrained(args_cli.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args_cli.model_id,
        dtype=torch.bfloat16,        # Better stability than fp16 on A100
        attn_implementation="sdpa", # Massive speedup for long reasoning traces
        device_map="auto",
    )

    dataset = load_dataset("codingmonster1234/chess-reasoning-processed")

    # 2. Optimized Training Arguments for 80GB VRAM
    training_args = SFTConfig(
        output_dir=f"./output-{args_cli.model_id.split('/')[-1]}",
        
        # Throughput Optimization
        bf16=True,                         # Uses A100 Tensor Cores efficiently
        tf32=True,                         # Math speedup for remaining float32 ops
        per_device_train_batch_size=2,    # Increased for 0.6B model on 80GB
        gradient_accumulation_steps=8,     # Minimal accumulation to keep throughput high
        gradient_checkpointing=True,      # Faster if you have enough VRAM (which you do)
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Modern practice
        dataloader_num_workers=4,          # Parallelize data loading
        
        # Evaluation & Logging
        eval_strategy="epoch",
        logging_steps=1,
        report_to="wandb",
        
        # Optimizer
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Checkpointing
        save_strategy="epoch",
        save_total_limit=2,
        
        max_length=2048,               # Adjust based on your trace lengths
        packing=False,                      # Pack multiple examples into one block for speed
    )

    # 3. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
    )

    trainer.train()
    
    # Final Eval
    metrics = trainer.evaluate()
    print(metrics)
    wandb.finish()

if __name__ == "__main__":
    main()