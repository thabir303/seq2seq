"""
Model 3: LSTM with Bahdanau Attention

Most advanced model using attention mechanism to overcome the
fixed-length context bottleneck.

Architecture:
    - Bidirectional LSTM Encoder: Encodes docstring with forward and backward context
    - Attention Mechanism: Bahdanau (additive) attention
    - LSTM Decoder with Attention: Attends to encoder outputs at each step

Goal:
    - Remove the fixed-context bottleneck
    - Improve code generation quality
    - Enable interpretability through attention visualization
"""

import torch
import torch.nn as nn
import random
from typing import Tuple, Optional, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    SOS_IDX, EOS_IDX, PAD_IDX, DEVICE, TEACHER_FORCING_RATIO
)
from .encoder import BidirectionalLSTMEncoder
from .decoder import AttentionDecoder


class LSTMAttentionSeq2Seq(nn.Module):
    """
    LSTM Sequence-to-Sequence Model with Bahdanau Attention.
    
    Components:
        - Bidirectional LSTM Encoder
        - Bahdanau (Additive) Attention
        - LSTM Decoder that attends to encoder outputs
        - Teacher forcing during training
    
    Key Difference:
        Instead of using only the final hidden state, the decoder
        attends to ALL encoder outputs at each step.
    """
    
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 embedding_dim: int = EMBEDDING_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT):
        """
        Initialize LSTM with Attention Seq2Seq model.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of layers
            dropout: Dropout probability
        """
        super(LSTMAttentionSeq2Seq, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_dim = hidden_dim
        
        # Bidirectional Encoder (output dim is 2 * hidden_dim)
        self.encoder = BidirectionalLSTMEncoder(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Decoder with Attention
        # Encoder hidden dim is 2x because of bidirectional
        self.decoder = AttentionDecoder(
            vocab_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            encoder_hidden_dim=hidden_dim * 2,  # Bidirectional
            decoder_hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.model_name = "lstm_attention"
    
    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create mask for source padding.
        
        Args:
            src: Source sequence [batch_size, src_len]
            
        Returns:
            mask: Boolean mask [batch_size, src_len] (True for valid positions)
        """
        return src != PAD_IDX
    
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_lengths: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = TEACHER_FORCING_RATIO) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_lengths: Actual source lengths [batch_size]
            teacher_forcing_ratio: Probability of using ground truth as input
            
        Returns:
            outputs: Output logits [batch_size, tgt_len, tgt_vocab_size]
            attentions: Attention weights [batch_size, tgt_len, src_len]
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        src_len = src.shape[1]
        tgt_vocab_size = self.tgt_vocab_size
        
        # Tensor to store decoder outputs and attention weights
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        attentions = torch.zeros(batch_size, tgt_len, src_len).to(src.device)
        
        # Create source mask
        mask = self.create_mask(src)
        
        # Encode source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        # First input to decoder is SOS token
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Decode step by step
        for t in range(1, tgt_len):
            # Decoder forward step with attention
            output, (hidden, cell), attention = self.decoder(
                decoder_input, (hidden, cell), encoder_outputs, mask
            )
            
            # Store outputs
            outputs[:, t, :] = output
            attentions[:, t, :attention.shape[1]] = attention
            
            # Decide whether to use teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(dim=1).unsqueeze(1)
        
        return outputs, attentions
    
    def generate(self,
                 src: torch.Tensor,
                 src_lengths: Optional[torch.Tensor] = None,
                 max_len: int = 80,
                 sos_idx: int = SOS_IDX,
                 eos_idx: int = EOS_IDX,
                 return_attention: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate output sequence without teacher forcing (inference mode).
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Actual source lengths [batch_size]
            max_len: Maximum output length
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: Generated token indices [batch_size, generated_len]
            outputs: Output logits [batch_size, generated_len, vocab_size]
            attentions: Attention weights [batch_size, generated_len, src_len] (if return_attention)
        """
        self.eval()
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        with torch.no_grad():
            # Create mask
            mask = self.create_mask(src)
            
            # Encode source
            encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
            
            # Start with SOS token
            decoder_input = torch.LongTensor([[sos_idx]] * batch_size).to(src.device)
            
            predictions = []
            output_probs = []
            all_attentions = [] if return_attention else None
            
            for _ in range(max_len):
                # Decoder step with attention
                output, (hidden, cell), attention = self.decoder(
                    decoder_input, (hidden, cell), encoder_outputs, mask
                )
                output_probs.append(output.unsqueeze(1))
                
                if return_attention:
                    all_attentions.append(attention.unsqueeze(1))
                
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
            
            if return_attention:
                attentions = torch.cat(all_attentions, dim=1)
            else:
                attentions = None
        
        return predictions, outputs, attentions
    
    def get_attention_weights(self,
                              src: torch.Tensor,
                              tgt: torch.Tensor,
                              src_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights for a given source-target pair.
        Used for visualization.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_lengths: Actual source lengths [batch_size]
            
        Returns:
            attentions: Attention weights [batch_size, tgt_len, src_len]
        """
        self.eval()
        with torch.no_grad():
            _, attentions = self.forward(src, tgt, src_lengths, teacher_forcing_ratio=0.0)
        return attentions
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_attention_model(src_vocab_size: int,
                           tgt_vocab_size: int,
                           device: torch.device = DEVICE) -> LSTMAttentionSeq2Seq:
    """
    Factory function to create and initialize LSTM with Attention model.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    model = LSTMAttentionSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"LSTM with Attention Seq2Seq Model")
    print(f"  Trainable parameters: {model.count_parameters():,}")
    print(f"  Embedding dim: {EMBEDDING_DIM}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Encoder output dim: {HIDDEN_DIM * 2} (bidirectional)")
    print(f"  Num layers: {NUM_LAYERS}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_attention_model(
        src_vocab_size=5000,
        tgt_vocab_size=5000,
        device=DEVICE
    )
    
    # Test forward pass
    batch_size = 4
    src_len = 20
    tgt_len = 30
    
    src = torch.randint(1, 5000, (batch_size, src_len)).to(DEVICE)  # Avoid padding idx 0
    tgt = torch.randint(1, 5000, (batch_size, tgt_len)).to(DEVICE)
    src_lengths = torch.LongTensor([20, 18, 15, 10])
    
    outputs, attentions = model(src, tgt, src_lengths)
    print(f"\nOutput shape: {outputs.shape}")
    print(f"Attention shape: {attentions.shape}")
    
    # Test generation
    predictions, _, gen_attentions = model.generate(src, src_lengths, max_len=30)
    print(f"Generated shape: {predictions.shape}")
    print(f"Generation attention shape: {gen_attentions.shape}")
