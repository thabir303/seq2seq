"""
Evaluation Script for Seq2Seq Text-to-Python Code Generation.

Evaluates all trained models on the test set using:
    - Token-level Accuracy
    - BLEU Score
    - Exact Match Accuracy

Also generates sample outputs and error analysis.

Usage:
    python evaluate.py --model vanilla_rnn
    python evaluate.py --model lstm
    python evaluate.py --model lstm_attention
    python evaluate.py --model all
"""

import os
import sys
import argparse
import json
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, PAD_IDX, SOS_IDX, EOS_IDX,
    CHECKPOINT_DIR, RESULTS_DIR, VISUALIZATION_DIR,
    MODEL_VANILLA_RNN, MODEL_LSTM, MODEL_LSTM_ATTENTION,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE, BATCH_SIZE, MAX_CODE_LENGTH
)
from data import get_dataloaders, Vocabulary
from models.vanilla_rnn import create_vanilla_rnn_model
from models.lstm_seq2seq import create_lstm_model
from models.lstm_attention import create_attention_model
from utils.metrics import (
    calculate_bleu, calculate_accuracy, calculate_exact_match,
    calculate_syntax_accuracy, analyze_errors, calculate_metrics_by_length
)
from utils.helpers import (
    load_checkpoint, save_results, tokens_to_string
)
from utils.visualization import (
    plot_metrics_comparison, plot_performance_by_length
)


def generate_predictions(model: torch.nn.Module,
                         data_loader: torch.utils.data.DataLoader,
                         tgt_vocab: Vocabulary,
                         max_len: int = MAX_CODE_LENGTH,
                         device: torch.device = DEVICE) -> Tuple[List, List, List, List]:
    """
    Generate predictions for all examples in data loader.
    
    Args:
        model: Trained model
        data_loader: Data loader
        tgt_vocab: Target vocabulary
        max_len: Maximum output length
        device: Device to use
        
    Returns:
        Tuple of (predictions, references, pred_tokens, ref_tokens)
    """
    model.eval()
    
    all_predictions = []
    all_references = []
    all_pred_tokens = []
    all_ref_tokens = []
    all_src_lengths = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating predictions"):
            src = batch['source'].to(device)
            tgt = batch['target'].to(device)
            src_lengths = batch['source_length']
            target_tokens = batch['target_tokens']
            
            # Generate predictions
            if model.model_name == 'lstm_attention':
                predictions, _, _ = model.generate(
                    src, src_lengths, max_len=max_len,
                    sos_idx=SOS_IDX, eos_idx=EOS_IDX
                )
            else:
                predictions, _ = model.generate(
                    src, src_lengths, max_len=max_len,
                    sos_idx=SOS_IDX, eos_idx=EOS_IDX
                )
            
            # Convert to tokens
            batch_size = predictions.shape[0]
            for i in range(batch_size):
                # Predicted tokens
                pred_indices = predictions[i].cpu().tolist()
                pred_tokens = tgt_vocab.indices_to_tokens(pred_indices)
                
                # Reference tokens (ground truth)
                ref_tokens = target_tokens[i]
                
                all_predictions.append(pred_indices)
                all_references.append(batch['target'][i].cpu().tolist())
                all_pred_tokens.append(pred_tokens)
                all_ref_tokens.append(ref_tokens)
                all_src_lengths.append(src_lengths[i].item())
    
    return all_predictions, all_references, all_pred_tokens, all_ref_tokens, all_src_lengths


def evaluate_model(model: torch.nn.Module,
                   test_loader: torch.utils.data.DataLoader,
                   tgt_vocab: Vocabulary,
                   model_name: str,
                   device: torch.device = DEVICE) -> Dict:
    """
    Comprehensive evaluation of a single model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        tgt_vocab: Target vocabulary
        model_name: Name of the model
        device: Device to use
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()} Model")
    print(f"{'='*60}")
    
    # Generate predictions
    predictions, references, pred_tokens, ref_tokens, src_lengths = generate_predictions(
        model, test_loader, tgt_vocab, device=device
    )
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # BLEU Score
    bleu_score = calculate_bleu(ref_tokens, pred_tokens)
    print(f"  BLEU Score: {bleu_score:.4f}")
    
    # Token Accuracy
    token_accuracy = calculate_accuracy(predictions, references, pad_idx=PAD_IDX)
    print(f"  Token Accuracy: {token_accuracy:.4f}")
    
    # Exact Match
    ref_strings = [tokens_to_string(t) for t in ref_tokens]
    pred_strings = [tokens_to_string(t) for t in pred_tokens]
    exact_match = calculate_exact_match(ref_strings, pred_strings)
    print(f"  Exact Match: {exact_match:.4f}")
    
    # Syntax Accuracy (bonus)
    syntax_accuracy, error_types = calculate_syntax_accuracy(pred_strings)
    print(f"  Syntax Accuracy: {syntax_accuracy:.4f}")
    print(f"  Error Types: {error_types}")
    
    # Error Analysis
    error_analysis = analyze_errors(ref_tokens, pred_tokens)
    print(f"  Missing Rate: {error_analysis['missing_rate']:.4f}")
    print(f"  Extra Rate: {error_analysis['extra_rate']:.4f}")
    print(f"  Wrong Rate: {error_analysis['wrong_rate']:.4f}")
    
    # Performance by length
    metrics_by_length = calculate_metrics_by_length(
        ref_tokens, pred_tokens, src_lengths
    )
    
    # Plot performance by length
    plot_performance_by_length(
        metrics_by_length,
        metric_name='bleu',
        save_path=os.path.join(VISUALIZATION_DIR, f'{model_name}_performance_by_length.png')
    )
    
    # Compile results
    results = {
        'model': model_name,
        'bleu_score': bleu_score,
        'token_accuracy': token_accuracy,
        'exact_match': exact_match,
        'syntax_accuracy': syntax_accuracy,
        'error_types': error_types,
        'error_analysis': error_analysis,
        'metrics_by_length': {k: dict(v) for k, v in metrics_by_length.items()}
    }
    
    # Save results
    save_results(results, f'{model_name}_evaluation.json')
    
    # Print sample predictions
    print("\n--- Sample Predictions ---")
    for i in range(min(3, len(pred_tokens))):
        print(f"\nExample {i+1}:")
        print(f"  Reference: {' '.join(ref_tokens[i][:20])}...")
        print(f"  Predicted: {' '.join(pred_tokens[i][:20])}...")
    
    return results


def load_model_for_evaluation(model_type: str,
                              src_vocab_size: int,
                              tgt_vocab_size: int,
                              device: torch.device = DEVICE) -> torch.nn.Module:
    """
    Load trained model for evaluation.
    
    Args:
        model_type: Model type
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        device: Device to use
        
    Returns:
        Loaded model
    """
    # Create model
    if model_type == MODEL_VANILLA_RNN:
        model = create_vanilla_rnn_model(src_vocab_size, tgt_vocab_size, device)
    elif model_type == MODEL_LSTM:
        model = create_lstm_model(src_vocab_size, tgt_vocab_size, device)
    elif model_type == MODEL_LSTM_ATTENTION:
        model = create_attention_model(src_vocab_size, tgt_vocab_size, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    try:
        load_checkpoint(
            model,
            checkpoint_path=None,
            model_name=model_type,
            load_best=True,
            device=device
        )
        print(f"Loaded best checkpoint for {model_type}")
    except FileNotFoundError:
        print(f"WARNING: No checkpoint found for {model_type}, using random weights")
    
    return model


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Seq2Seq models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['vanilla_rnn', 'lstm', 'lstm_attention', 'all'],
                        help='Model to evaluate (default: all)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Seq2Seq Model Evaluation")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load data (we only need test set)
    print("\nLoading test data...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=args.batch_size,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE
    )
    
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Determine which models to evaluate
    if args.model == 'all':
        models_to_evaluate = [MODEL_VANILLA_RNN, MODEL_LSTM, MODEL_LSTM_ATTENTION]
    else:
        models_to_evaluate = [args.model]
    
    # Evaluate models
    all_results = {}
    
    for model_type in models_to_evaluate:
        # Load model
        model = load_model_for_evaluation(
            model_type, len(src_vocab), len(tgt_vocab), DEVICE
        )
        
        # Evaluate
        results = evaluate_model(
            model, test_loader, tgt_vocab, model_type, DEVICE
        )
        
        all_results[model_type] = results
    
    # Compare all models if evaluating all
    if args.model == 'all' and len(all_results) > 0:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Create comparison table
        print("\n{:<20} {:>12} {:>12} {:>12}".format(
            "Model", "BLEU", "Token Acc", "Exact Match"
        ))
        print("-"*60)
        
        comparison_metrics = {}
        for model_type, results in all_results.items():
            print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format(
                model_type,
                results['bleu_score'],
                results['token_accuracy'],
                results['exact_match']
            ))
            
            comparison_metrics[model_type] = {
                'bleu': results['bleu_score'],
                'token_accuracy': results['token_accuracy'],
                'exact_match': results['exact_match'],
                'syntax_accuracy': results['syntax_accuracy']
            }
        
        # Plot comparison
        plot_metrics_comparison(
            comparison_metrics,
            save_path=os.path.join(VISUALIZATION_DIR, 'model_comparison.png')
        )
        
        # Save combined results
        save_results(all_results, 'all_models_evaluation.json')
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
