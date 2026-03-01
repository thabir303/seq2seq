"""
Model 4: Transformer Seq2Seq

Transformer-based model for comparison with RNN/LSTM models.
Uses self-attention mechanism instead of recurrence.

Architecture:
    - Transformer Encoder: Multi-head self-attention + feedforward layers
    - Transformer Decoder: Masked multi-head self-attention + cross-attention + feedforward
    - Positional Encoding: Sinusoidal positional embeddings

Goal:
    - Compare Transformer architecture against RNN-based seq2seq models
    - Demonstrate the power of self-attention for sequence tasks
    - Provide a modern baseline for the code generation task
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    SOS_IDX, EOS_IDX, PAD_IDX, DEVICE, TEACHER_FORCING_RATIO,
    MAX_CODE_LENGTH
)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    Adds positional information to embeddings since Transformers
    have no inherent notion of sequence order.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        
        Args:
            x: Embedded input [batch_size, seq_len, d_model]
        Returns:
            Positionally encoded tensor [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    """
    Transformer Sequence-to-Sequence Model.
    
    Components:
        - Token Embeddings + Positional Encoding (source & target)
        - Transformer Encoder (multi-head self-attention + FFN)
        - Transformer Decoder (masked self-attention + cross-attention + FFN)
        - Linear output projection
    
    Key Differences from RNN-based models:
        - No recurrence — relies entirely on attention
        - Parallel processing of all positions during training
        - Positional encoding to inject order information
        - Multi-head attention for richer representations
    """
    
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = EMBEDDING_DIM,
                 nhead: int = 8,
                 num_encoder_layers: int = NUM_LAYERS,
                 num_decoder_layers: int = NUM_LAYERS,
                 dim_feedforward: int = HIDDEN_DIM,
                 dropout: float = DROPOUT,
                 max_len: int = 512):
        """
        Initialize Transformer Seq2Seq model.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            d_model: Dimension of embeddings / model hidden size
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_len: Maximum sequence length for positional encoding
        """
        super(TransformerSeq2Seq, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        
        # Source embeddings + positional encoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.src_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Target embeddings + positional encoding
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Scaling factor for embeddings
        self.scale = math.sqrt(d_model)
        
        self.model_name = "transformer"
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask for decoder (upper triangular = -inf).
        
        Args:
            sz: Sequence length
            device: Device
            
        Returns:
            mask: [sz, sz] with -inf above diagonal
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_lengths: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = TEACHER_FORCING_RATIO) -> torch.Tensor:
        """
        Forward pass through the Transformer model.
        
        Note: teacher_forcing_ratio is accepted for API compatibility but
        Transformers always use teacher forcing during training (parallel decoding).
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_lengths: Actual source lengths (unused, for API compat)
            teacher_forcing_ratio: Unused (Transformers use full teacher forcing)
            
        Returns:
            outputs: Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Create masks
        src_key_padding_mask = (src == PAD_IDX)  # [batch_size, src_len]
        tgt_key_padding_mask = (tgt == PAD_IDX)  # [batch_size, tgt_len]
        
        tgt_len = tgt.shape[1]
        tgt_mask = self._generate_square_subsequent_mask(tgt_len, src.device)
        
        # Embed and add positional encoding
        src_emb = self.src_pos_encoding(self.src_embedding(src) * self.scale)
        tgt_emb = self.tgt_pos_encoding(self.tgt_embedding(tgt) * self.scale)
        
        # Transformer forward
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.fc_out(output)  # [batch_size, tgt_len, tgt_vocab_size]
        
        return logits
    
    def generate(self,
                 src: torch.Tensor,
                 src_lengths: Optional[torch.Tensor] = None,
                 max_len: int = 80,
                 sos_idx: int = SOS_IDX,
                 eos_idx: int = EOS_IDX) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate output sequence autoregressively (inference mode).
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Actual source lengths (unused, for API compat)
            max_len: Maximum output length
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
            
        Returns:
            predictions: Generated token indices [batch_size, generated_len]
            outputs: Output logits [batch_size, generated_len, vocab_size]
        """
        self.eval()
        batch_size = src.shape[0]
        device = src.device
        
        with torch.no_grad():
            # Encode source once
            src_key_padding_mask = (src == PAD_IDX)
            src_emb = self.src_pos_encoding(self.src_embedding(src) * self.scale)
            memory = self.transformer.encoder(
                src_emb,
                src_key_padding_mask=src_key_padding_mask
            )
            
            # Start with SOS token
            tgt_indices = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
            
            predictions = []
            output_probs = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for _ in range(max_len):
                # Create target mask
                tgt_len = tgt_indices.shape[1]
                tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)
                tgt_key_padding_mask = (tgt_indices == PAD_IDX)
                
                # Embed target
                tgt_emb = self.tgt_pos_encoding(self.tgt_embedding(tgt_indices) * self.scale)
                
                # Decode
                dec_output = self.transformer.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                
                # Get last token's logits
                last_logits = self.fc_out(dec_output[:, -1, :])  # [batch_size, vocab_size]
                output_probs.append(last_logits.unsqueeze(1))
                
                # Greedy decode
                pred_token = last_logits.argmax(dim=-1)  # [batch_size]
                predictions.append(pred_token.unsqueeze(1))
                
                # Update finished status
                finished = finished | (pred_token == eos_idx)
                
                # Append predicted token to decoder input
                tgt_indices = torch.cat([tgt_indices, pred_token.unsqueeze(1)], dim=1)
                
                # Stop if all sequences have generated EOS
                if finished.all():
                    break
            
            predictions = torch.cat(predictions, dim=1)   # [batch_size, gen_len]
            outputs = torch.cat(output_probs, dim=1)       # [batch_size, gen_len, vocab_size]
        
        return predictions, outputs
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer_model(src_vocab_size: int,
                              tgt_vocab_size: int,
                              device: torch.device = DEVICE) -> TransformerSeq2Seq:
    """
    Factory function to create and initialize Transformer Seq2Seq model.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=EMBEDDING_DIM,
        nhead=8,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        dim_feedforward=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)
    
    print(f"Transformer Seq2Seq Model")
    print(f"  Trainable parameters: {model.count_parameters():,}")
    print(f"  d_model: {EMBEDDING_DIM}")
    print(f"  Attention heads: 8")
    print(f"  Encoder layers: {NUM_LAYERS}")
    print(f"  Decoder layers: {NUM_LAYERS}")
    print(f"  FFN dim: {HIDDEN_DIM}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_transformer_model(
        src_vocab_size=5000,
        tgt_vocab_size=5000,
        device=DEVICE
    )
    
    # Test forward pass
    batch_size = 4
    src_len = 20
    tgt_len = 30
    
    src = torch.randint(1, 5000, (batch_size, src_len)).to(DEVICE)
    tgt = torch.randint(1, 5000, (batch_size, tgt_len)).to(DEVICE)
    src_lengths = torch.LongTensor([20, 18, 15, 10])
    
    outputs = model(src, tgt, src_lengths)
    print(f"\nOutput shape: {outputs.shape}")
    
    # Test generation
    predictions, _ = model.generate(src, src_lengths, max_len=30)
    print(f"Generated shape: {predictions.shape}")
