import argparse
import wandb
import torch
import re
import numpy as np
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, EvalPrediction
import chess


def preprocess_logits_for_metrics(logits, labels):
    """
    This runs on the GPU. We convert raw logits to the most likely token IDs.
    Logits: (batch_size, sequence_length, vocab_size)
    Returns: (batch_size, sequence_length)
    """
    if isinstance(logits, tuple):
        # Depending on the model, logits might be the first element of a tuple
        logits = logits[0]
    return logits.argmax(dim=-1)

class MoveMetricAccumulator:
    def __init__(self, tokenizer:PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        self.total_correct = 0
        self.total_examples = 0
        self.total_tokens = 0

    def __call__(self, eval_prediction: EvalPrediction, compute_result: bool = False):
        if not compute_result:
            # --- BATCH LEVEL ACCUMULATION ---
            # IMPORTANT: eval_prediction.predictions is now already the argmax result (Token IDs)
            # thanks to preprocess_logits_for_metrics.
            pred_ids = eval_prediction.predictions
            labels = eval_prediction.label_ids
            inputs = eval_prediction.inputs

            # Ensure labels ignore padding/masked values (-100)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

            # 2. Decode Batch
            decoded_inputs = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)
            decoded_preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, label in zip(decoded_preds, decoded_labels):
                self.total_examples += 1
                
                # Track generated length
                self.total_tokens += len(self.tokenizer.encode(pred))

                # Move Matching logic
                match = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL)
                pred_move = match.group(1).strip() if match else ""
                
                # We check if the predicted move string matches the one in the label
                if pred_move and pred_move in label:
                    self.total_correct += 1
            
            return {} 

        else:
            # --- FINAL SUMMARY CALCULATION ---
            metrics = {
                "eval_move_accuracy": self.total_correct / max(1, self.total_examples),
                "eval_avg_token_length": self.total_tokens / max(1, self.total_examples),
            }
            self.reset() 
            return metrics


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

    tokenizer = AutoTokenizer.from_pretrained(args_cli.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args_cli.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )

    dataset = load_dataset("codingmonster1234/chess-reasoning-processed")

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
        report_to="wandb",
        batch_eval_metrics=True, # Required for the compute_result logic
        
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        save_strategy="epoch",
        save_total_limit=2,
        
        max_length=2048,
        packing=False,
    )

    move_metrics_callable = MoveMetricAccumulator(tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        compute_metrics=move_metrics_callable,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    initial_metrics = trainer.evaluate()
    print("Initial Evaluation Metrics:", initial_metrics)

    trainer.train()
    
    final_metrics = trainer.evaluate()
    print("Final Evaluation Metrics:", final_metrics)
    
    wandb.finish()

if __name__ == "__main__":
    main()