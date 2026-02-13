"""
Helper utilities for training and evaluation.

Contains:
    - Checkpoint saving and loading
    - Parameter counting
    - Logging utilities
    - Code generation helpers
"""

import os
import sys
import json
import torch
from typing import Dict, Any, Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, RESULTS_DIR


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    train_loss: float,
                    val_loss: float,
                    metrics: Dict[str, float],
                    model_name: str,
                    checkpoint_dir: str = CHECKPOINT_DIR,
                    is_best: bool = False):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_name': model_name,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metrics': metrics
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_latest.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")
    
    # Save epoch-specific checkpoint periodically
    if epoch % 5 == 0:
        epoch_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch}.pt')
        torch.save(checkpoint, epoch_path)


def load_checkpoint(model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    checkpoint_path: str = None,
                    model_name: str = None,
                    load_best: bool = True,
                    checkpoint_dir: str = CHECKPOINT_DIR,
                    device: torch.device = None) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        checkpoint_path: Full path to checkpoint file
        model_name: Model name (used if checkpoint_path not provided)
        load_best: Whether to load best model (True) or latest (False)
        checkpoint_dir: Directory containing checkpoints
        device: Device to load model to
        
    Returns:
        Dictionary with epoch, losses, and metrics
    """
    if checkpoint_path is None:
        suffix = 'best' if load_best else 'latest'
        checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_{suffix}.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    return {
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'metrics': checkpoint.get('metrics', {})
    }


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_results(results: Dict[str, Any],
                 filename: str,
                 results_dir: str = RESULTS_DIR):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary of results
        filename: Output filename
        results_dir: Directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {filepath}")


def load_results(filename: str,
                 results_dir: str = RESULTS_DIR) -> Dict[str, Any]:
    """
    Load evaluation results from JSON file.
    
    Args:
        filename: Input filename
        results_dir: Directory containing results
        
    Returns:
        Dictionary of results
    """
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def tokens_to_string(tokens: List[str]) -> str:
    """
    Convert token list to code string.
    
    Handles special tokens like NEWLINE and INDENT.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Code string
    """
    code = ' '.join(tokens)
    
    # Replace special tokens
    code = code.replace(' <NEWLINE> ', '\n')
    code = code.replace('<NEWLINE>', '\n')
    code = code.replace(' <INDENT> ', '    ')
    code = code.replace('<INDENT>', '    ')
    
    return code


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


class TrainingLogger:
    """
    Logger for tracking training progress.
    """
    
    def __init__(self, model_name: str, log_dir: str = RESULTS_DIR):
        """
        Initialize logger.
        
        Args:
            model_name: Name of the model
            log_dir: Directory to save logs
        """
        self.model_name = model_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  metrics: Optional[Dict[str, float]] = None):
        """
        Log information for one epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Optional metrics dictionary
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        if metrics:
            self.metrics_history.append({'epoch': epoch, **metrics})
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if metrics:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            print(f"  Metrics: {metrics_str}")
    
    def save_logs(self):
        """Save training logs to file."""
        logs = {
            'model_name': self.model_name,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }
        
        save_results(logs, f'{self.model_name}_training_log.json', self.log_dir)
    
    def get_best_epoch(self) -> int:
        """Get epoch with lowest validation loss."""
        return int(torch.tensor(self.val_losses).argmin().item()) + 1
