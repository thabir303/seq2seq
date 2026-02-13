"""
Decoder implementations for Seq2Seq models.
Contains: RNNDecoder, LSTMDecoder, AttentionDecoder
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .attention import BahdanauAttention


class RNNDecoder(nn.Module):
    """
    Vanilla RNN Decoder.
    
    Generates target sequence one token at a time using the context vector
    from the encoder.
    
    Architecture:
        Embedding + Context -> RNN -> Dense -> Output Token
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        """
        Initialize RNN Decoder.
        
        Args:
            vocab_size: Size of target vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of RNN hidden state
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super(RNNDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                input_token: torch.Tensor,
                hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one decoding step.
        
        Args:
            input_token: Current input token [batch_size, 1]
            hidden: Previous hidden state [num_layers, batch_size, hidden_dim]
            
        Returns:
            output: Token probabilities [batch_size, vocab_size]
            hidden: New hidden state [num_layers, batch_size, hidden_dim]
        """
        # Embed input token: [batch_size, 1, embedding_dim]
        embedded = self.dropout(self.embedding(input_token))
        
        # RNN forward
        rnn_output, hidden = self.rnn(embedded, hidden)
        
        # Output layer: [batch_size, vocab_size]
        output = self.fc_out(rnn_output.squeeze(1))
        
        return output, hidden


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder.
    
    Uses LSTM for better long-range dependency handling.
    
    Architecture:
        Embedding -> LSTM -> Dense -> Output Token
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        """
        Initialize LSTM Decoder.
        
        Args:
            vocab_size: Size of target vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                input_token: torch.Tensor,
                hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for one decoding step.
        
        Args:
            input_token: Current input token [batch_size, 1]
            hidden: Previous (hidden, cell) states
            
        Returns:
            output: Token probabilities [batch_size, vocab_size]
            (hidden, cell): New hidden and cell states
        """
        # Embed input token
        embedded = self.dropout(self.embedding(input_token))
        
        # LSTM forward
        lstm_output, (hidden, cell) = self.lstm(embedded, hidden)
        
        # Output layer
        output = self.fc_out(lstm_output.squeeze(1))
        
        return output, (hidden, cell)


class AttentionDecoder(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention.
    
    At each step, attends to encoder outputs to create context vector,
    which is concatenated with the embedding before LSTM.
    
    Architecture:
        Embedding + Attention(encoder_outputs) -> LSTM -> Dense -> Output Token
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 encoder_hidden_dim: int,
                 decoder_hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        """
        Initialize Attention Decoder.
        
        Args:
            vocab_size: Size of target vocabulary
            embedding_dim: Dimension of embeddings
            encoder_hidden_dim: Dimension of encoder hidden states (2x for bidirectional)
            decoder_hidden_dim: Dimension of decoder LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(AttentionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_layers = num_layers
        
        # Attention mechanism
        self.attention = BahdanauAttention(encoder_hidden_dim, decoder_hidden_dim)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer (input: embedding + context)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer (combines LSTM output, context, and embedding)
        self.fc_out = nn.Linear(decoder_hidden_dim + encoder_hidden_dim + embedding_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                input_token: torch.Tensor,
                hidden: Tuple[torch.Tensor, torch.Tensor],
                encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass for one decoding step with attention.
        
        Args:
            input_token: Current input token [batch_size, 1]
            hidden: Previous (hidden, cell) states
            encoder_outputs: All encoder outputs [batch_size, src_len, encoder_hidden_dim]
            mask: Source mask [batch_size, src_len]
            
        Returns:
            output: Token probabilities [batch_size, vocab_size]
            (hidden, cell): New hidden and cell states
            attention_weights: Attention weights [batch_size, src_len]
        """
        # Embed input token: [batch_size, 1, embedding_dim]
        embedded = self.dropout(self.embedding(input_token))
        embedded = embedded.squeeze(1)  # [batch_size, embedding_dim]
        
        # Get previous hidden state for attention
        # Use top layer hidden state
        prev_hidden = hidden[0][-1]  # [batch_size, decoder_hidden_dim]
        
        # Compute attention
        context, attention_weights = self.attention(prev_hidden, encoder_outputs, mask)
        
        # Concatenate embedding and context
        lstm_input = torch.cat([embedded, context], dim=1)  # [batch_size, embedding_dim + encoder_hidden]
        lstm_input = lstm_input.unsqueeze(1)  # [batch_size, 1, embedding_dim + encoder_hidden]
        
        # LSTM forward
        lstm_output, (new_hidden, new_cell) = self.lstm(lstm_input, hidden)
        lstm_output = lstm_output.squeeze(1)  # [batch_size, decoder_hidden_dim]
        
        # Combine LSTM output, context, and embedding for final prediction
        combined = torch.cat([lstm_output, context, embedded], dim=1)
        output = self.fc_out(combined)  # [batch_size, vocab_size]
        
        return output, (new_hidden, new_cell), attention_weights
