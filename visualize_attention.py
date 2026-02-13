"""
Attention Visualization Script for LSTM with Attention model.

Generates attention heatmaps for test examples showing:
    - Alignment between docstring tokens and generated code tokens
    - Which source words the model focuses on for each output token

Required for the attention analysis section of the assignment.

Usage:
    python visualize_attention.py --num_examples 5
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, CHECKPOINT_DIR, VISUALIZATION_DIR,
    MODEL_LSTM_ATTENTION, SOS_IDX, EOS_IDX,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE, BATCH_SIZE, MAX_CODE_LENGTH
)
from data import get_dataloaders, Vocabulary
from models.lstm_attention import create_attention_model
from utils.helpers import load_checkpoint
from utils.visualization import plot_attention, plot_multiple_attentions


def get_attention_examples(model: torch.nn.Module,
                           data_loader: torch.utils.data.DataLoader,
                           src_vocab: Vocabulary,
                           tgt_vocab: Vocabulary,
                           num_examples: int = 5,
                           device: torch.device = DEVICE) -> List[Dict]:
    """
    Get attention examples for visualization.
    
    Args:
        model: Trained LSTM with Attention model
        data_loader: Data loader
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        num_examples: Number of examples to extract
        device: Device to use
        
    Returns:
        List of dictionaries with attention, source, and target tokens
    """
    model.eval()
    examples = []
    
    with torch.no_grad():
        for batch in data_loader:
            if len(examples) >= num_examples:
                break
            
            src = batch['source'].to(device)
            tgt = batch['target'].to(device)
            src_lengths = batch['source_length']
            source_tokens = batch['source_tokens']
            target_tokens = batch['target_tokens']
            
            # Generate predictions with attention
            predictions, _, attentions = model.generate(
                src, src_lengths, max_len=MAX_CODE_LENGTH,
                sos_idx=SOS_IDX, eos_idx=EOS_IDX,
                return_attention=True
            )
            
            # Extract examples from batch
            batch_size = predictions.shape[0]
            for i in range(batch_size):
                if len(examples) >= num_examples:
                    break
                
                # Get attention for this example
                attention = attentions[i].cpu().numpy()  # [generated_len, src_len]
                
                # Get source tokens
                src_tokens = source_tokens[i]
                
                # Get generated tokens
                pred_indices = predictions[i].cpu().tolist()
                gen_tokens = tgt_vocab.indices_to_tokens(pred_indices)
                
                # Trim to actual lengths
                src_len = min(len(src_tokens), attention.shape[1])
                gen_len = min(len(gen_tokens), attention.shape[0])
                
                attention = attention[:gen_len, :src_len]
                src_tokens = src_tokens[:src_len]
                gen_tokens = gen_tokens[:gen_len]
                
                # Ground truth for comparison
                gt_tokens = target_tokens[i]
                
                examples.append({
                    'attention': attention,
                    'source': src_tokens,
                    'target': gen_tokens,
                    'ground_truth': gt_tokens
                })
    
    return examples


def analyze_attention(examples: List[Dict]) -> None:
    """
    Analyze attention patterns and print insights.
    
    Args:
        examples: List of attention examples
    """
    print("\n" + "="*60)
    print("ATTENTION ANALYSIS")
    print("="*60)
    
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i+1} ---")
        print(f"Source (docstring): {' '.join(ex['source'][:15])}...")
        print(f"Generated: {' '.join(ex['target'][:15])}...")
        print(f"Ground truth: {' '.join(ex['ground_truth'][:15])}...")
        
        # Analyze attention patterns
        attention = ex['attention']
        source = ex['source']
        target = ex['target']
        
        # Find keywords that receive high attention
        print("\nHigh attention alignments:")
        
        for t_idx, tgt_token in enumerate(target[:10]):
            if t_idx >= attention.shape[0]:
                break
            
            # Get attention weights for this target token
            attn_weights = attention[t_idx]
            
            # Find top attending source tokens
            top_k = min(3, len(source))
            top_indices = np.argsort(attn_weights)[-top_k:][::-1]
            
            if len(top_indices) > 0:
                top_src = [source[idx] if idx < len(source) else "?" for idx in top_indices]
                top_weights = [attn_weights[idx] for idx in top_indices if idx < len(source)]
                
                if top_weights:
                    attention_str = ", ".join([f"{s}({w:.2f})" for s, w in zip(top_src, top_weights)])
                    print(f"  '{tgt_token}' <- {attention_str}")
        
        # Check for semantic alignments
        print("\nSemantic alignment check:")
        
        # Common keywords to look for
        keywords = ['maximum', 'minimum', 'return', 'add', 'remove', 'get', 'set', 'list', 'string']
        
        for kw in keywords:
            if kw in [s.lower() for s in source]:
                src_idx = [s.lower() for s in source].index(kw)
                
                # Find which target tokens attend to this keyword
                attn_to_kw = attention[:, src_idx] if src_idx < attention.shape[1] else []
                
                if len(attn_to_kw) > 0:
                    max_attn_idx = np.argmax(attn_to_kw)
                    if max_attn_idx < len(target):
                        max_attn_token = target[max_attn_idx]
                        max_attn_val = attn_to_kw[max_attn_idx]
                        print(f"  '{kw}' most attended by '{max_attn_token}' (weight: {max_attn_val:.3f})")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize attention weights')
    parser.add_argument('--num_examples', type=int, default=5,
                        help='Number of examples to visualize (default: 5)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for data loading')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Attention Visualization")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Number of examples: {args.num_examples}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=args.batch_size,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE
    )
    
    # Create and load model
    print("\nLoading LSTM with Attention model...")
    model = create_attention_model(
        len(src_vocab), len(tgt_vocab), DEVICE
    )
    
    try:
        load_checkpoint(
            model,
            model_name=MODEL_LSTM_ATTENTION,
            load_best=True,
            device=DEVICE
        )
    except FileNotFoundError:
        print("WARNING: No checkpoint found, using random weights")
        print("Train the model first using: python train.py --model lstm_attention")
    
    # Get attention examples
    print("\nExtracting attention examples...")
    examples = get_attention_examples(
        model, test_loader, src_vocab, tgt_vocab,
        num_examples=args.num_examples, device=DEVICE
    )
    
    print(f"Extracted {len(examples)} examples")
    
    # Create visualization directory
    attn_dir = os.path.join(VISUALIZATION_DIR, 'attention')
    os.makedirs(attn_dir, exist_ok=True)
    
    # Generate attention heatmaps
    print("\nGenerating attention heatmaps...")
    
    for i, ex in enumerate(examples):
        save_path = os.path.join(attn_dir, f'attention_example_{i+1}.png')
        
        # Limit size for visualization
        max_src = min(30, len(ex['source']))
        max_tgt = min(40, len(ex['target']))
        
        attention = ex['attention'][:max_tgt, :max_src]
        source = ex['source'][:max_src]
        target = ex['target'][:max_tgt]
        
        plot_attention(
            attention_weights=attention,
            source_tokens=source,
            target_tokens=target,
            title=f'Attention Visualization - Example {i+1}',
            save_path=save_path,
            figsize=(14, 10)
        )
        
        print(f"  Saved: {save_path}")
    
    # Analyze attention patterns
    analyze_attention(examples)
    
    # Save example data for report
    examples_data = []
    for i, ex in enumerate(examples):
        examples_data.append({
            'example_id': i + 1,
            'source': ex['source'],
            'generated': ex['target'],
            'ground_truth': ex['ground_truth'],
            'attention_shape': list(ex['attention'].shape)
        })
    
    import json
    with open(os.path.join(attn_dir, 'attention_examples.json'), 'w') as f:
        json.dump(examples_data, f, indent=2)
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print(f"Heatmaps saved to: {attn_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
