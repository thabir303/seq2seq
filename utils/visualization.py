"""
Visualization utilities for Seq2Seq models.

Contains:
    - Training/validation loss curve plotting
    - Attention heatmap visualization
    - Performance vs docstring length plots
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VISUALIZATION_DIR


def plot_loss_curves(train_losses: List[float],
                     val_losses: List[float],
                     model_name: str,
                     save_path: Optional[str] = None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        model_name: Name of the model for title
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} - Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Mark minimum validation loss
    min_val_idx = np.argmin(val_losses)
    plt.scatter([min_val_idx + 1], [val_losses[min_val_idx]], 
                color='red', s=100, zorder=5, marker='*',
                label=f'Best Val Loss: {val_losses[min_val_idx]:.4f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curves to {save_path}")
    
    plt.close()


def plot_all_models_loss(all_losses: Dict[str, Dict[str, List[float]]],
                         save_path: Optional[str] = None):
    """
    Plot loss curves for all models on the same figure.
    
    Args:
        all_losses: Dictionary {model_name: {'train': [...], 'val': [...]}}
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'vanilla_rnn': 'blue', 'lstm': 'green', 'lstm_attention': 'red'}
    labels = {'vanilla_rnn': 'Vanilla RNN', 'lstm': 'LSTM', 'lstm_attention': 'LSTM + Attention'}
    
    # Training Loss
    ax1 = axes[0]
    for model_name, losses in all_losses.items():
        epochs = range(1, len(losses['train']) + 1)
        color = colors.get(model_name, 'purple')
        label = labels.get(model_name, model_name)
        ax1.plot(epochs, losses['train'], color=color, label=label, linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Validation Loss
    ax2 = axes[1]
    for model_name, losses in all_losses.items():
        epochs = range(1, len(losses['val']) + 1)
        color = colors.get(model_name, 'purple')
        label = labels.get(model_name, model_name)
        ax2.plot(epochs, losses['val'], color=color, label=label, linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison curves to {save_path}")
    
    plt.close()


def plot_attention(attention_weights: np.ndarray,
                   source_tokens: List[str],
                   target_tokens: List[str],
                   title: str = "Attention Weights",
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (12, 8)):
    """
    Plot attention heatmap showing alignment between source and target tokens.
    
    Args:
        attention_weights: Attention matrix [tgt_len, src_len]
        source_tokens: List of source (docstring) tokens
        target_tokens: List of target (code) tokens
        title: Title for the plot
        save_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Ensure dimensions match
    attn = attention_weights[:len(target_tokens), :len(source_tokens)]
    
    # Create heatmap
    ax = sns.heatmap(
        attn,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cmap='Blues',
        annot=False,
        fmt='.2f',
        cbar=True,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.xlabel('Source (Docstring)', fontsize=12)
    plt.ylabel('Target (Generated Code)', fontsize=12)
    plt.title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")
    
    plt.close()


def plot_multiple_attentions(attention_list: List[Dict],
                             save_dir: str = VISUALIZATION_DIR):
    """
    Plot multiple attention heatmaps.
    
    Args:
        attention_list: List of dicts with 'attention', 'source', 'target' keys
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, item in enumerate(attention_list):
        attention = item['attention']
        source = item['source']
        target = item['target']
        
        save_path = os.path.join(save_dir, f'attention_example_{i+1}.png')
        
        plot_attention(
            attention_weights=attention,
            source_tokens=source,
            target_tokens=target,
            title=f'Attention Visualization - Example {i+1}',
            save_path=save_path
        )


def plot_performance_by_length(metrics_by_length: Dict[str, Dict[str, float]],
                               metric_name: str = 'bleu',
                               save_path: Optional[str] = None):
    """
    Plot performance metric vs docstring length.
    
    Args:
        metrics_by_length: Dictionary from calculate_metrics_by_length
        metric_name: Name of metric to plot
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    # Sort bins by their starting number
    bins = sorted(metrics_by_length.keys(), 
                  key=lambda x: int(x.split('-')[0].replace('+', '')))
    
    values = [metrics_by_length[b][metric_name] for b in bins]
    counts = [metrics_by_length[b].get('count', 0) for b in bins]
    
    # Create bar plot
    x = range(len(bins))
    bars = plt.bar(x, values, color='steelblue', edgecolor='black', alpha=0.8)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count}',
                ha='center', va='bottom', fontsize=9)
    
    plt.xticks(x, bins, fontsize=10)
    plt.xlabel('Docstring Length (tokens)', fontsize=12)
    plt.ylabel(metric_name.upper(), fontsize=12)
    plt.title(f'{metric_name.upper()} Score vs Docstring Length', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved performance plot to {save_path}")
    
    plt.close()


def plot_metrics_comparison(metrics: Dict[str, Dict[str, float]],
                            save_path: Optional[str] = None):
    """
    Plot bar chart comparing metrics across different models.
    
    Args:
        metrics: Dictionary {model_name: {metric: value}}
        save_path: Path to save figure
    """
    models = list(metrics.keys())
    metric_names = list(metrics[models[0]].keys())
    
    n_models = len(models)
    n_metrics = len(metric_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    labels = {'vanilla_rnn': 'Vanilla RNN', 'lstm': 'LSTM', 'lstm_attention': 'LSTM + Attention'}
    
    for i, model in enumerate(models):
        values = [metrics[model][m] for m in metric_names]
        label = labels.get(model, model)
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=colors[i % len(colors)])
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization utilities...")
    
    # Test loss curves
    train_losses = [2.5, 2.0, 1.7, 1.5, 1.3, 1.2, 1.1, 1.05, 1.0, 0.95]
    val_losses = [2.6, 2.1, 1.8, 1.6, 1.5, 1.45, 1.42, 1.40, 1.38, 1.37]
    
    plot_loss_curves(
        train_losses, val_losses, 
        "Test Model",
        save_path=os.path.join(VISUALIZATION_DIR, "test_loss_curves.png")
    )
    
    # Test attention plot
    np.random.seed(42)
    attention = np.random.rand(10, 8)
    attention = attention / attention.sum(axis=1, keepdims=True)  # Normalize rows
    
    source = ["returns", "the", "maximum", "value", "from", "a", "list", "of"]
    target = ["def", "max_val", "(", "nums", ")", ":", "return", "max", "(", "nums"]
    
    plot_attention(
        attention, source, target,
        title="Test Attention Heatmap",
        save_path=os.path.join(VISUALIZATION_DIR, "test_attention.png")
    )
    
    print("Visualization tests completed!")
