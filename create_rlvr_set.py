import bisect
from datasets import load_dataset, DatasetDict, concatenate_datasets
import chess
import os

def transform_row(example):
    board = chess.Board(example['fen'])
    example['uci_moves'] = example['moves'].strip().split(" ")
    # Using .pop is cleaner than del for map functions
    example.pop('moves', None)
    example['tags'] = example['tags'].strip().split(" ")
    for key in ['board', 'white_kingside', 'white_queenside', 'black_kingside', 'black_queenside']:
        example.pop(key, None)
    example['turn'] = 'White' if board.turn == chess.WHITE else 'Black'
    return example

def process_and_upload_chess_dataset(source_dataset_path, repo_id):
    ds = load_dataset(source_dataset_path, split='train')
    
    print("Processing rows...")
    ds = ds.map(transform_row, num_proc=os.cpu_count(), desc="Transforming rows")
    
    print("Sorting by rating...")
    ds = ds.sort("rating")
    
    # Extract ratings into a list for fast binary search
    # This is much faster than calling ds['rating'] repeatedly in the loop
    all_ratings = ds['rating']
    
    min_rating, max_rating, step = 400, 3300, 100
    train_splits, val_splits, test_splits = [], [], []
    
    print("Bucketing and splitting data via binary search...")
    
    for start_val in range(min_rating, max_rating, step):
        end_val = start_val + step
        
        # Binary search to find index range [left, right)
        left_idx = bisect.bisect_left(all_ratings, start_val)
        right_idx = bisect.bisect_left(all_ratings, end_val)
        
        # Check if bucket has data
        if left_idx == right_idx:
            continue
            
        # Select the slice instantaneously
        bucket = ds.select(range(left_idx, right_idx))
            
        # Stratified splits
        train_test_split = bucket.train_test_split(test_size=0.15, seed=42)
        temp_split = train_test_split['test'].train_test_split(test_size=0.333, seed=42)
        
        train_splits.append(train_test_split['train'])
        val_splits.append(temp_split['train']) 
        test_splits.append(temp_split['test'])

    final_ds = DatasetDict({
        'train': concatenate_datasets(train_splits),
        'validation': concatenate_datasets(val_splits),
        'test': concatenate_datasets(test_splits)
    })
    
    print(f"Final Counts - Train: {len(final_ds['train'])}, Val: {len(final_ds['validation'])}, Test: {len(final_ds['test'])}")
    
    final_ds.push_to_hub(repo_id)
    print(f"Dataset uploaded to {repo_id}")

# Usage:
process_and_upload_chess_dataset("codingmonster1234/chess_puzzles_dataset", "codingmonster1234/chess-puzzles-rlvr")