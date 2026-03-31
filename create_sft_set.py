import json
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from prompts import system_prompt, build_user_prompt, build_completion

def process_split(dataset, split_name, color="#00ff00"):
    """
    Processes a dataset split and returns a list of dictionaries.
    """
    processed_data = []
    
    for example in tqdm(dataset, desc=f"Processing {split_name}", colour=color):
        # Constructing the chat-style format
        prompt = [
            {"content": system_prompt, "role": "system"},
            {"content": build_user_prompt(example["fen"]), "role": "user"}
        ]
        
        completion = [
            {"content": build_completion(example["answer"], example["first_move"]), "role": "assistant"}
        ]
        
        processed_data.append({
            "prompt": prompt, 
            "completion": completion
        })
    
    # Convert list of dicts to a Hugging Face Dataset object
    return Dataset.from_list(processed_data)

# 1. Load the raw datasets
print("Loading source datasets...")
raw_train = load_dataset("pilipolio/chess-reasoning-traces", split="train")
raw_test = load_dataset("pilipolio/chess-reasoning-traces", split="test")

# 2. Process each split into HF Dataset objects
train_ds = process_split(raw_train, "Train Split", color="#2ecc71")
test_ds = process_split(raw_test, "Test Split", color="#3498db")

# 3. Combine into a DatasetDict to preserve split names
dataset_dict = DatasetDict({
    "train": train_ds,
    "test": test_ds
})


# 4. Push to the Hub
# Replace 'your-username' with your actual HF handle
repo_id = "codingmonster1234/chess-reasoning-sft"
print(f"Uploading dataset to {repo_id}...")

dataset_dict.push_to_hub(repo_id)

print("\nUpload complete! You can view your dataset at:")
print(f"https://huggingface.co/datasets/{repo_id}")