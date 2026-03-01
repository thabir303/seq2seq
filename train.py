"""
Training Script for Seq2Seq Text-to-Python Code Generation.

This script trains all three models:
    1. Vanilla RNN Seq2Seq
    2. LSTM Seq2Seq
    3. LSTM with Attention

Uses the configuration from config.py and saves checkpoints to checkpoints/.

Usage:
    python train.py --model vanilla_rnn
    python train.py --model lstm
    python train.py --model lstm_attention
    python train.py --model all  # Train all models
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple, Dict, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, CLIP_GRAD,
    TEACHER_FORCING_RATIO, PAD_IDX, LOG_INTERVAL,
    MODEL_VANILLA_RNN, MODEL_LSTM, MODEL_LSTM_ATTENTION, MODEL_TRANSFORMER,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE, BATCH_SIZE
)
from data import get_dataloaders 
from models.vanilla_rnn import create_vanilla_rnn_model
from models.lstm_seq2seq import create_lstm_model
from models.lstm_attention import create_attention_model
from models.transformer_seq2seq import create_transformer_model
from utils.helpers import save_checkpoint, load_checkpoint, TrainingLogger, format_time
from utils.visualization import plot_loss_curves


def train_epoch(model: nn.Module,
                data_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                clip: float = CLIP_GRAD,
                teacher_forcing_ratio: float = TEACHER_FORCING_RATIO,
                device: torch.device = DEVICE) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        data_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        clip: Gradient clipping value
        teacher_forcing_ratio: Probability of using ground truth
        device: Device to use
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(data_loader, desc="Training", leave=False):
        # Get batch data
        src = batch['source'].to(device)
        tgt = batch['target'].to(device)
        src_lengths = batch['source_length']
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if model.model_name == 'lstm_attention':
            output, _ = model(src, tgt, src_lengths, teacher_forcing_ratio)
        elif model.model_name == 'transformer':
            output = model(src, tgt, src_lengths, teacher_forcing_ratio)
        else:
            output = model(src, tgt, src_lengths, teacher_forcing_ratio)
        
        # Reshape for loss calculation
        # output: [batch_size, tgt_len, vocab_size]
        # tgt: [batch_size, tgt_len]
        output_dim = output.shape[-1]
        
        # Ignore the first token (SOS) in target
        output = output[:, 1:].contiguous().view(-1, output_dim)
        tgt = tgt[:, 1:].contiguous().view(-1)
        
        # Calculate loss
        loss = criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping - to avoid exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)


def evaluate(model: nn.Module,
             data_loader: torch.utils.data.DataLoader,
             criterion: nn.Module,
             device: torch.device = DEVICE) -> float:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Model to evaluate
        data_loader: Validation/test data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        Average validation loss
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            src = batch['source'].to(device)
            tgt = batch['target'].to(device)
            src_lengths = batch['source_length']
            
            # Forward pass (no teacher forcing during evaluation)
            if model.model_name == 'lstm_attention':
                output, _ = model(src, tgt, src_lengths, teacher_forcing_ratio=0.0)
            elif model.model_name == 'transformer':
                output = model(src, tgt, src_lengths, teacher_forcing_ratio=0.0)
            else:
                output = model(src, tgt, src_lengths, teacher_forcing_ratio=0.0)
            
            # Reshape for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)


def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                num_epochs: int = NUM_EPOCHS,
                learning_rate: float = LEARNING_RATE,
                device: torch.device = DEVICE,
                resume: bool = False) -> Tuple[nn.Module, Dict]:
    """
    Train a single model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use
        resume: Whether to resume from checkpoint
        
    Returns:
        Trained model and training history
    """
    model_name = model.model_name
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} Model")
    print(f"{'='*60}")
    
    # Loss function (ignore padding)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler: reduce LR when val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Logger
    logger = TrainingLogger(model_name)
    
    # Track best model
    best_val_loss = float('inf')
    start_epoch = 1
    
    # Resume from checkpoint if requested
    if resume:
        from config import CHECKPOINT_DIR
        import os
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_latest.pt')
        if os.path.exists(checkpoint_path):
            checkpoint_info = load_checkpoint(
                model, optimizer, 
                model_name=model_name, 
                load_best=False,  # Load latest, not best
                device=device
            )
            start_epoch = checkpoint_info['epoch'] + 1
            best_val_loss = checkpoint_info['metrics'].get('best_val_loss', float('inf'))
            print(f"\nResuming training from epoch {start_epoch}")
            print(f"Best validation loss so far: {best_val_loss:.4f}")
        else:
            print(f"\nNo checkpoint found for {model_name}, starting from scratch.")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device=device
        )
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device=device)
        
        epoch_time = time.time() - epoch_start
        
        # Log progress
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Time:       {format_time(epoch_time)}")
        
        logger.log_epoch(epoch, train_loss, val_loss)
        
        # Step the learning rate scheduler
        scheduler.step(val_loss)
        
        # Check if best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  *** New best model! ***")
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            train_loss, val_loss,
            metrics={'best_val_loss': best_val_loss},
            model_name=model_name,
            is_best=is_best
        )
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save training logs
    logger.save_logs()
    
    # Plot loss curves (only if there are losses to plot)
    from config import VISUALIZATION_DIR
    if logger.train_losses and logger.val_losses:
        plot_loss_curves(
            logger.train_losses,
            logger.val_losses,
            model_name,
            save_path=os.path.join(VISUALIZATION_DIR, f'{model_name}_loss_curves.png')
        )
    else:
        print(f"No new training epochs — skipping loss curve plot for {model_name}.")
    
    return model, {
        'train_losses': logger.train_losses,
        'val_losses': logger.val_losses,
        'best_val_loss': best_val_loss
    }


def create_model(model_type: str,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 device: torch.device = DEVICE) -> nn.Module:
    """
    Create model based on type.
    
    Args:
        model_type: 'vanilla_rnn', 'lstm', or 'lstm_attention'
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        device: Device to place model on
        
    Returns:
        Created model
    """
    if model_type == MODEL_VANILLA_RNN:
        return create_vanilla_rnn_model(src_vocab_size, tgt_vocab_size, device)
    elif model_type == MODEL_LSTM:
        return create_lstm_model(src_vocab_size, tgt_vocab_size, device)
    elif model_type == MODEL_LSTM_ATTENTION:
        return create_attention_model(src_vocab_size, tgt_vocab_size, device)
    elif model_type == MODEL_TRANSFORMER:
        return create_transformer_model(src_vocab_size, tgt_vocab_size, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Seq2Seq models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['vanilla_rnn', 'lstm', 'lstm_attention', 'transformer', 'all'],
                        help='Model to train (default: all)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--train_size', type=int, default=TRAIN_SIZE,
                        help=f'Training set size (default: {TRAIN_SIZE})')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Seq2Seq Text-to-Python Code Generation")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Training Size: {args.train_size}")
    print("="*60)
    
    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE
    )
    
    print(f"\nSource Vocabulary Size: {len(src_vocab)}")
    print(f"Target Vocabulary Size: {len(tgt_vocab)}")
    
    # Save vocabularies
    from config import CHECKPOINT_DIR
    src_vocab.save(os.path.join(CHECKPOINT_DIR, 'src_vocab.pkl'))
    tgt_vocab.save(os.path.join(CHECKPOINT_DIR, 'tgt_vocab.pkl'))
    
    # Determine which models to train
    if args.model == 'all':
        models_to_train = [MODEL_VANILLA_RNN, MODEL_LSTM, MODEL_LSTM_ATTENTION, MODEL_TRANSFORMER]
    else:
        models_to_train = [args.model]
    
    # Train models
    all_histories = {}
    
    for model_type in models_to_train:
        # Create model
        model = create_model(
            model_type,
            len(src_vocab),
            len(tgt_vocab),
            DEVICE
        )
        
        # Train model
        model, history = train_model(
            model, train_loader, val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=DEVICE,
            resume=args.resume
        )
        
        all_histories[model_type] = history
    
    # Plot comparison if training all models
    if args.model == 'all':
        from utils.visualization import plot_all_models_loss
        from config import VISUALIZATION_DIR
        
        loss_data = {
            name: {'train': hist['train_losses'], 'val': hist['val_losses']}
            for name, hist in all_histories.items()
        }
        
        plot_all_models_loss(
            loss_data,
            save_path=os.path.join(VISUALIZATION_DIR, 'all_models_loss_comparison.png')
        )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Print summary
    print("\nFinal Results:")
    for model_type, history in all_histories.items():
        print(f"  {model_type}: Best Val Loss = {history['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
