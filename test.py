import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import system_prompt, build_user_prompt
from datasets import load_dataset
from rich import print_json

model_id = "codingmonster1234/chess-sft-model"

# 1. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa" # Keeps it fast without the version mismatch error
)

test_set = load_dataset("codingmonster1234/chess-reasoning-processed", split="test")
example = test_set[0]
print_json(example)


inputs = tokenizer(example, return_tensors="pt").to("cuda")

# 3. Generate the Reasoning + Move
output = model.generate(
    **inputs,
    max_new_tokens=2048, # Enough space for the <think> block
    do_sample=True,      # Set to False for "Greedy" (more deterministic) moves
    temperature=0,
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(output[0], skip_special_tokens=True))