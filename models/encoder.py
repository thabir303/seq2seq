"""
Encoder implementations for Seq2Seq models.
Contains: RNNEncoder, LSTMEncoder, BidirectionalLSTMEncoder
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional


class RNNEncoder(nn.Module):
    """
    Vanilla RNN Encoder.
    
    Encodes source sequence into a fixed-length context vector.
    
    Architecture:
        Embedding -> RNN -> Final Hidden State (context vector)
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        """
        Initialize RNN Encoder.
        
        Args:
            vocab_size: Size of source vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of RNN hidden state
            num_layers: Number of RNN layers
            dropout: Dropout rate (applied if num_layers > 1)
        """
        super(RNNEncoder, self).__init__()
        
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
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                src: torch.Tensor,
                src_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            src: Source sequence [batch_size, seq_len]
            src_lengths: Actual lengths of source sequences [batch_size]
            
        Returns:
            outputs: All RNN outputs [batch_size, seq_len, hidden_dim]
            hidden: Final hidden state [num_layers, batch_size, hidden_dim]
        """
        # Embed source tokens
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, embedding_dim]
        
        # Pack if lengths provided (for efficiency)
        if src_lengths is not None:
            # Ensure lengths are on CPU and sorted
            src_lengths = src_lengths.cpu()
            packed = pack_padded_sequence(embedded, src_lengths, batch_first=True, enforce_sorted=True)
            outputs, hidden = self.rnn(packed)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, hidden = self.rnn(embedded)
        
        return outputs, hidden


class LSTMEncoder(nn.Module):
    """
    LSTM Encoder.
    
    Better at capturing long-range dependencies than vanilla RNN.
    
    Architecture:
        Embedding -> LSTM -> Final Hidden State + Cell State (context)
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        """
        Initialize LSTM Encoder.
        
        Args:
            vocab_size: Size of source vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMEncoder, self).__init__()
        
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
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                src: torch.Tensor,
                src_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through encoder.
        
        Args:
            src: Source sequence [batch_size, seq_len]
            src_lengths: Actual lengths of source sequences [batch_size]
            
        Returns:
            outputs: All LSTM outputs [batch_size, seq_len, hidden_dim]
            (hidden, cell): Final hidden and cell states [num_layers, batch_size, hidden_dim]
        """
        # Embed source tokens
        embedded = self.dropout(self.embedding(src))
        
        # Pack if lengths provided
        if src_lengths is not None:
            src_lengths = src_lengths.cpu()
            packed = pack_padded_sequence(embedded, src_lengths, batch_first=True, enforce_sorted=True)
            outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        
        return outputs, (hidden, cell)


class BidirectionalLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    
    Processes sequence in both directions for richer representations.
    Used in attention-based models.
    
    Architecture:
        Embedding -> BiLSTM -> Outputs + Combined Hidden State
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        """
        Initialize Bidirectional LSTM Encoder.
        
        Args:
            vocab_size: Size of source vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of LSTM hidden state (in each direction)
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(BidirectionalLSTMEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Linear layers to combine bidirectional states
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                src: torch.Tensor,
                src_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through encoder.
        
        Args:
            src: Source sequence [batch_size, seq_len]
            src_lengths: Actual lengths of source sequences [batch_size]
            
        Returns:
            outputs: All BiLSTM outputs [batch_size, seq_len, hidden_dim * 2]
            (hidden, cell): Combined final states [num_layers, batch_size, hidden_dim]
        """
        # Embed source tokens
        embedded = self.dropout(self.embedding(src))
        
        # Pack if lengths provided
        if src_lengths is not None:
            src_lengths = src_lengths.cpu()
            packed = pack_padded_sequence(embedded, src_lengths, batch_first=True, enforce_sorted=True)
            outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        
        # Combine forward and backward hidden states
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        # -> [num_layers, batch_size, hidden_dim * 2]
        # -> [num_layers, batch_size, hidden_dim]
        
        batch_size = src.shape[0]
        
        # Reshape hidden: separate forward and backward
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        hidden = torch.tanh(self.fc_hidden(hidden))
        
        # Reshape cell
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        cell = torch.tanh(self.fc_cell(cell))
        
        return outputs, (hidden, cell)
