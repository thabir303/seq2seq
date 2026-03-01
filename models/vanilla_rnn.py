"""
Model 1: Vanilla RNN Seq2Seq

This is the baseline model for the assignment.
Uses simple RNN cells for both encoder and decoder.

Architecture:
    - RNN Encoder: Encodes docstring into fixed-length context vector
    - RNN Decoder: Generates Python code from context vector

Goal:
    - Establish baseline performance
    - Observe performance degradation for longer sequences
"""

import torch
import torch.nn as nn
import random
from typing import Tuple, Dict, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    SOS_IDX, EOS_IDX, DEVICE, TEACHER_FORCING_RATIO
)
from .encoder import RNNEncoder
from .decoder import RNNDecoder


class VanillaRNNSeq2Seq(nn.Module):
    """
    Vanilla RNN Sequence-to-Sequence Model.
    
    Components:
        - RNN Encoder
        - RNN Decoder
        - Teacher forcing during training
    
    The encoder final hidden state serves as the fixed-length context vector
    that initializes the decoder.
    """
    
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 embedding_dim: int = EMBEDDING_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT):
        """
        Initialize Vanilla RNN Seq2Seq model.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of RNN hidden states
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super(VanillaRNNSeq2Seq, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = RNNEncoder(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = RNNDecoder(
            vocab_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.model_name = "vanilla_rnn"
    
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_lengths: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = TEACHER_FORCING_RATIO) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_lengths: Actual source lengths [batch_size]
            teacher_forcing_ratio: Probability of using ground truth as input
            
        Returns:
            outputs: Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.tgt_vocab_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        
        # Encode source sequence
        _, hidden = self.encoder(src, src_lengths)
        
        # First input to decoder is SOS token
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Decode step by step
        for t in range(1, tgt_len):
            # Decoder forward step
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Store output (prediction)
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                # Use ground truth as next input
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                # Use predicted token as next input
                decoder_input = output.argmax(dim=1).unsqueeze(1)
        
        return outputs
    
    def generate(self,
                 src: torch.Tensor,
                 src_lengths: Optional[torch.Tensor] = None,
                 max_len: int = 80,
                 sos_idx: int = SOS_IDX,
                 eos_idx: int = EOS_IDX) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate output sequence without teacher forcing (inference mode).
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Actual source lengths [batch_size]
            max_len: Maximum output length
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
            
        Returns:
            predictions: Generated token indices [batch_size, generated_len]
            outputs: Output logits [batch_size, generated_len, vocab_size]
        """
        self.eval()
        batch_size = src.shape[0]
        
        with torch.no_grad():
            # Encode source
            _, hidden = self.encoder(src, src_lengths)
            
            # Start with SOS token
            decoder_input = torch.LongTensor([[sos_idx]] * batch_size).to(src.device)
            
            predictions = []
            output_probs = []
            
            for _ in range(max_len):
                # Decoder step
                output, hidden = self.decoder(decoder_input, hidden)
                output_probs.append(output.unsqueeze(1))
                
                # Get predicted token
                pred_token = output.argmax(dim=1)
                predictions.append(pred_token.unsqueeze(1))
                
                # Update input for next step
                decoder_input = pred_token.unsqueeze(1)
                
                # Stop if all sequences have generated EOS
                if (pred_token == eos_idx).all():
                    break
            
            predictions = torch.cat(predictions, dim=1)
            outputs = torch.cat(output_probs, dim=1)
        
        return predictions, outputs
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_vanilla_rnn_model(src_vocab_size: int,
                             tgt_vocab_size: int,
                             device: torch.device = DEVICE) -> VanillaRNNSeq2Seq:
    """
    Factory function to create and initialize Vanilla RNN Seq2Seq model.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    model = VanillaRNNSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"Vanilla RNN Seq2Seq Model")
    print(f"  Trainable parameters: {model.count_parameters():,}")
    print(f"  Embedding dim: {EMBEDDING_DIM}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Num layers: {NUM_LAYERS}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_vanilla_rnn_model(
        src_vocab_size=5000,
        tgt_vocab_size=5000,
        device=DEVICE
    )
    
    # Test forward pass
    batch_size = 4
    src_len = 20
    tgt_len = 30
    
    src = torch.randint(0, 5000, (batch_size, src_len)).to(DEVICE)
    tgt = torch.randint(0, 5000, (batch_size, tgt_len)).to(DEVICE)
    src_lengths = torch.LongTensor([20, 18, 15, 10])
    
    outputs = model(src, tgt, src_lengths)
    print(f"\nOutput shape: {outputs.shape}")
    
    # Test generation
    predictions, _ = model.generate(src, src_lengths, max_len=30)
    print(f"Generated shape: {predictions.shape}")
