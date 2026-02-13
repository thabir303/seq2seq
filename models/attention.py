"""
Bahdanau (Additive) Attention mechanism.
Used in the LSTM with Attention model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism.
    
    Computes attention weights and context vector based on:
        - Decoder hidden state (query)
        - Encoder outputs (keys/values)
    
    Score function: v^T * tanh(W_h * h + W_s * s)
    
    Where:
        - h: encoder hidden states
        - s: decoder hidden state
        - W_h, W_s: learnable weight matrices
        - v: learnable weight vector
    """
    
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int):
        """
        Initialize attention layer.
        
        Args:
            encoder_hidden_dim: Dimension of encoder hidden states
            decoder_hidden_dim: Dimension of decoder hidden state
        """
        super(BahdanauAttention, self).__init__()
        
        # For bidirectional encoder, dimensions are doubled
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        
        # Learnable parameters
        self.W_h = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.W_s = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
    
    def forward(self,
                hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            hidden: Decoder hidden state [batch_size, decoder_hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, encoder_hidden_dim]
            mask: Source mask [batch_size, src_len] (True for valid positions)
            
        Returns:
            context: Context vector [batch_size, encoder_hidden_dim]
            attention_weights: Attention weights [batch_size, src_len]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Project encoder outputs: [batch_size, src_len, decoder_hidden_dim]
        encoder_proj = self.W_h(encoder_outputs)
        
        # Project decoder hidden: [batch_size, decoder_hidden_dim]
        # Expand for broadcasting: [batch_size, 1, decoder_hidden_dim]
        decoder_proj = self.W_s(hidden).unsqueeze(1)
        
        # Compute attention scores: [batch_size, src_len, 1]
        energy = torch.tanh(encoder_proj + decoder_proj)
        attention_scores = self.v(energy).squeeze(2)  # [batch_size, src_len]
        
        # Apply mask (set padding positions to -inf)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, src_len]
        
        # Compute context vector (weighted sum of encoder outputs)
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, src_len]
            encoder_outputs                   # [batch_size, src_len, encoder_hidden_dim]
        ).squeeze(1)  # [batch_size, encoder_hidden_dim]
        
        return context, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    This is an optional extension - can be used for bonus points
    if comparing with a Transformer-based model.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_dim: Dimension of hidden state
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = self.head_dim ** 0.5
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor [batch_size, query_len, hidden_dim]
            key: Key tensor [batch_size, key_len, hidden_dim]
            value: Value tensor [batch_size, value_len, hidden_dim]
            mask: Attention mask [batch_size, query_len, key_len]
            
        Returns:
            output: Attended output [batch_size, query_len, hidden_dim]
            attention_weights: Attention weights [batch_size, num_heads, query_len, key_len]
        """
        batch_size = query.shape[0]
        
        # Project queries, keys, values
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Final projection
        output = self.W_o(context)
        
        return output, attention_weights
