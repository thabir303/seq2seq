"""
Dataset loading and preprocessing for CodeSearchNet Python dataset.
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_NAME, TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
    MAX_DOCSTRING_LENGTH, MAX_CODE_LENGTH,
    PAD_IDX, SOS_IDX, EOS_IDX, BATCH_SIZE, DEVICE
)
from data.vocabulary import Vocabulary, tokenize, tokenize_code


class CodeSearchNetDataset(Dataset):
    """
    PyTorch Dataset for CodeSearchNet Python dataset.
    
    Each example contains:
        - source: Tokenized and indexed docstring
        - target: Tokenized and indexed Python code
        - source_length: Actual length of source sequence
        - target_length: Actual length of target sequence
    """
    
    def __init__(self, 
                 data: List[Dict],
                 src_vocab: Vocabulary,
                 tgt_vocab: Vocabulary,
                 max_src_len: int = MAX_DOCSTRING_LENGTH,
                 max_tgt_len: int = MAX_CODE_LENGTH):
        """
        Initialize dataset.
        
        Args:
            data: List of data examples with 'docstring' and 'code' keys
            src_vocab: Source vocabulary for docstrings
            tgt_vocab: Target vocabulary for code
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
        """
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Preprocess all examples
        self.examples = self._preprocess()
    
    def _preprocess(self) -> List[Dict]:
        """Preprocess all examples."""
        examples = []
        
        for item in tqdm(self.data, desc="Preprocessing data"):
            docstring = item.get('docstring', item.get('func_documentation_string', ''))
            code = item.get('code', item.get('func_code_string', ''))
            
            if not docstring or not code:
                continue
            
            # Tokenize
            src_tokens = tokenize(docstring)
            tgt_tokens = tokenize_code(code)
            
            # Truncate if necessary
            src_tokens = src_tokens[:self.max_src_len]
            tgt_tokens = tgt_tokens[:self.max_tgt_len - 2]  # Leave room for SOS and EOS
            
            # Convert to indices
            src_indices = self.src_vocab.tokens_to_indices(src_tokens)
            
            # Add SOS and EOS to target
            tgt_indices = [SOS_IDX] + self.tgt_vocab.tokens_to_indices(tgt_tokens) + [EOS_IDX]
            
            examples.append({
                'source': src_indices,
                'target': tgt_indices,
                'source_tokens': src_tokens,
                'target_tokens': tgt_tokens,
                'source_length': len(src_indices),
                'target_length': len(tgt_indices)
            })
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.
    Pads sequences to the same length within a batch.
    
    Args:
        batch: List of examples from dataset
        
    Returns:
        Dictionary with padded tensors
    """
    # Sort by source length (descending) for packed sequences
    batch = sorted(batch, key=lambda x: x['source_length'], reverse=True)
    
    # Extract sequences
    sources = [torch.LongTensor(item['source']) for item in batch]
    targets = [torch.LongTensor(item['target']) for item in batch]
    
    source_lengths = torch.LongTensor([item['source_length'] for item in batch])
    target_lengths = torch.LongTensor([item['target_length'] for item in batch])
    
    # Pad sequences
    sources_padded = pad_sequence(sources, batch_first=True, padding_value=PAD_IDX)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
    
    # Get original tokens for evaluation
    source_tokens = [item['source_tokens'] for item in batch]
    target_tokens = [item['target_tokens'] for item in batch]
    
    return {
        'source': sources_padded,
        'target': targets_padded,
        'source_length': source_lengths,
        'target_length': target_lengths,
        'source_tokens': source_tokens,
        'target_tokens': target_tokens
    }


def load_codesearchnet_data(split: str = 'train', 
                            size: Optional[int] = None) -> List[Dict]:
    """
    Load CodeSearchNet Python dataset from Hugging Face.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        size: Number of examples to load (None for all)
        
    Returns:
        List of data examples
    """
    print(f"Loading CodeSearchNet Python dataset ({split})...")
    
    try:
        # Try loading the dataset
        dataset = load_dataset(DATASET_NAME, split=split)
        
        if size:
            dataset = dataset.select(range(min(size, len(dataset))))
        
        # Convert to list of dicts
        data = [dict(item) for item in dataset]
        print(f"Loaded {len(data)} examples from {split} split")
        return data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting alternative loading method...")
        
        # Alternative: load all splits and select
        try:
            dataset = load_dataset(DATASET_NAME)
            
            if split in dataset:
                data = dataset[split]
            else:
                # If split not found, use train and split manually
                data = dataset['train']
            
            if size:
                data = data.select(range(min(size, len(data))))
            
            data = [dict(item) for item in data]
            print(f"Loaded {len(data)} examples")
            return data
            
        except Exception as e2:
            print(f"Failed to load dataset: {e2}")
            raise


def build_vocabularies(train_data: List[Dict],
                       min_freq: int = 2) -> Tuple[Vocabulary, Vocabulary]:
    """
    Build source and target vocabularies from training data.
    
    Args:
        train_data: Training data examples
        min_freq: Minimum token frequency
        
    Returns:
        Tuple of (source_vocab, target_vocab)
    """
    print("Building vocabularies...")
    
    # Collect all tokens
    src_texts = []
    tgt_texts = []
    
    for item in tqdm(train_data, desc="Tokenizing"):
        docstring = item.get('docstring', item.get('func_documentation_string', ''))
        code = item.get('code', item.get('func_code_string', ''))
        
        if docstring and code:
            src_texts.append(tokenize(docstring))
            tgt_texts.append(tokenize_code(code))
    
    # Build vocabularies
    src_vocab = Vocabulary(name="source")
    src_vocab.build_from_texts(src_texts, min_freq=min_freq)
    
    tgt_vocab = Vocabulary(name="target")
    tgt_vocab.build_from_texts(tgt_texts, min_freq=min_freq)
    
    return src_vocab, tgt_vocab


def get_dataloaders(batch_size: int = BATCH_SIZE,
                    train_size: int = TRAIN_SIZE,
                    val_size: int = VAL_SIZE,
                    test_size: int = TEST_SIZE) -> Tuple:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        batch_size: Batch size for DataLoaders
        train_size: Number of training examples
        val_size: Number of validation examples
        test_size: Number of test examples
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, src_vocab, tgt_vocab)
    """
    # Load data
    train_data = load_codesearchnet_data('train', train_size)
    
    # Try to load validation data, fallback to using part of train
    try:
        val_data = load_codesearchnet_data('validation', val_size)
    except:
        print("Validation split not available, using part of training data")
        val_data = train_data[-val_size:]
        train_data = train_data[:-val_size]
    
    # Try to load test data, fallback to using part of train
    try:
        test_data = load_codesearchnet_data('test', test_size)
    except:
        print("Test split not available, using part of training data")
        test_data = train_data[-test_size:]
        train_data = train_data[:-test_size]
    
    # Build vocabularies from training data
    src_vocab, tgt_vocab = build_vocabularies(train_data)
    
    # Create datasets
    train_dataset = CodeSearchNetDataset(train_data, src_vocab, tgt_vocab)
    val_dataset = CodeSearchNetDataset(val_data, src_vocab, tgt_vocab)
    test_dataset = CodeSearchNetDataset(test_data, src_vocab, tgt_vocab)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Training examples: {len(train_dataset)}")
    print(f"  Validation examples: {len(val_dataset)}")
    print(f"  Test examples: {len(test_dataset)}")
    print(f"  Source vocabulary size: {len(src_vocab)}")
    print(f"  Target vocabulary size: {len(tgt_vocab)}")
    
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=4,
        train_size=100,
        val_size=20,
        test_size=20
    )
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Source: {batch['source'].shape}")
    print(f"  Target: {batch['target'].shape}")
    
    # Print example
    print(f"\nExample source tokens: {batch['source_tokens'][0][:10]}...")
    print(f"Example target tokens: {batch['target_tokens'][0][:10]}...")
