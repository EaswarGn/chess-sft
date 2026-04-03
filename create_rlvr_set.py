from datasets import load_dataset, DatasetDict, concatenate_datasets
import chess

def transform_row(example):
    board = chess.Board(example['fen'])
    example['uci_moves'] = example['moves'].strip().split(" ")
    del example['moves'] 
    example['tags'] = example['tags'].strip().split(" ")
    del example['board']
    del example['white_kingside']
    del example['white_queenside']
    del example['black_kingside']  
    del example['black_queenside']
    example['turn'] = 'White' if board.turn == chess.WHITE else 'Black'
    return example

def process_and_upload_chess_dataset(source_dataset_path, repo_id):
    # Load the dataset (assuming it's the 'train' split initially)
    ds = load_dataset(source_dataset_path, split='train')
    
    
    print("Processing rows...")
    ds = ds.map(
        transform_row, 
        num_proc=4, # Use multiple CPU cores to speed this up
        desc="Transforming rows"
    )
    
    # Sort by rating to ensure organization
    ds = ds.sort("rating")
    
    # Define our rating boundaries
    min_rating = 400
    max_rating = 3300
    step = 100
    
    train_splits = []
    val_splits = []
    test_splits = []
    
    print("Bucketing and splitting data...")
    
    for start in range(min_rating, max_rating, step):
        end = start + step
        
        # Create the bucket for this rating range
        bucket = ds.filter(lambda x: start <= x['rating'] < end)
        
        if len(bucket) == 0:
            continue
            
        # First split: 85% Train vs 15% (Remainder)
        train_test_split = bucket.train_test_split(test_size=0.15, seed=42)
        
        # Second split: From the 15%, take 1/3 for Test (5% total) and 2/3 for Val (10% total)
        # 0.333 of 15% is approx 5%
        temp_split = train_test_split['test'].train_test_split(test_size=0.333, seed=42)
        
        train_splits.append(train_test_split['train'])
        val_splits.append(temp_split['train']) # This becomes the 10%
        test_splits.append(temp_split['test'])  # This becomes the 5%

    # 2. Combine all buckets back into unified splits
    final_ds = DatasetDict({
        'train': concatenate_datasets(train_splits),
        'validation': concatenate_datasets(val_splits),
        'test': concatenate_datasets(test_splits)
    })
    
    print(f"Final Counts - Train: {len(final_ds['train'])}, Val: {len(final_ds['validation'])}, Test: {len(final_ds['test'])}")
    
    # 3. Push to the Hub
    final_ds.push_to_hub(repo_id)
    print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_id}")

# Usage:
process_and_upload_chess_dataset("codingmonster1234/chess_puzzles_dataset", "codingmonster1234/chess-puzzles-rlvr")