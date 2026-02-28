"""
Vocabulary class for tokenization and index mapping.
Handles both source (docstring) and target (code) vocabularies.
"""

import pickle
import os
from collections import Counter
from typing import List, Dict, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from config import PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX


class Vocabulary:
    """
    Vocabulary class that handles token-to-index and index-to-token mappings.
    
    Attributes:
        token2idx: Dictionary mapping tokens to their indices
        idx2token: Dictionary mapping indices to their tokens
        token_counts: Counter tracking token frequencies
    """
    
    def __init__(self, name: str = "vocab"):
        """
        Initialize vocabulary with special tokens.
        
        Args:
            name: Name identifier for this vocabulary
        """
        self.name = name
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.token_counts: Counter = Counter()
        
        # Add special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [
            (PAD_TOKEN, PAD_IDX),
            (SOS_TOKEN, SOS_IDX),
            (EOS_TOKEN, EOS_IDX),
            (UNK_TOKEN, UNK_IDX),
        ]
        
        for token, idx in special_tokens:
            self.token2idx[token] = idx
            self.idx2token[idx] = token
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token2idx)
    
    def add_token(self, token: str) -> int:
        """
        Add a token to the vocabulary.
        
        Args:
            token: Token string to add
            
        Returns:
            Index of the token
        """
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        return self.token2idx[token]
    
    def add_tokens(self, tokens: List[str]):
        """
        Add multiple tokens to the vocabulary.
        
        Args:
            tokens: List of token strings to add
        """
        for token in tokens:
            self.add_token(token)
            self.token_counts[token] += 1
    
    def get_index(self, token: str) -> int:
        """
        Get index for a token.
        
        Args:
            token: Token string
            
        Returns:
            Token index, UNK_IDX if token not in vocabulary
        """
        return self.token2idx.get(token, UNK_IDX)
    
    def get_token(self, idx: int) -> str:
        """
        Get token for an index.
        
        Args:
            idx: Token index
            
        Returns:
            Token string, UNK_TOKEN if index not in vocabulary
        """
        return self.idx2token.get(idx, UNK_TOKEN)
    
    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens to indices.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token indices
        """
        return [self.get_index(token) for token in tokens]
    
    def indices_to_tokens(self, indices: List[int], 
                          remove_special: bool = True) -> List[str]:
        """
        Convert a list of indices to tokens.
        
        Args:
            indices: List of token indices
            remove_special: Whether to remove special tokens from output
            
        Returns:
            List of token strings
        """
        special_indices = {PAD_IDX, SOS_IDX, EOS_IDX}
        tokens = []
        
        for idx in indices:
            if idx == EOS_IDX:
                break
            if remove_special and idx in special_indices:
                continue
            tokens.append(self.get_token(idx))
        
        return tokens
    
    def build_from_texts(self, texts: List[List[str]], 
                         min_freq: int = 1,
                         max_size: Optional[int] = None):
        """
        Build vocabulary from a list of tokenized texts.
        
        Args:
            texts: List of tokenized texts (each text is a list of tokens)
            min_freq: Minimum frequency for a token to be included
            max_size: Maximum vocabulary size (None for unlimited)
        """
        # Count all tokens
        for tokens in texts:
            self.token_counts.update(tokens)
        
        # Filter by frequency and add to vocabulary
        sorted_tokens = sorted(
            self.token_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for token, count in sorted_tokens:
            if count < min_freq:
                break
            if max_size and len(self.token2idx) >= max_size:
                break
            self.add_token(token)
        
        print(f"Built {self.name} vocabulary with {len(self)} tokens")
    
    def save(self, path: str):
        """Save vocabulary to file."""
        data = {
            'name': self.name,
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'token_counts': dict(self.token_counts)
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved vocabulary to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(name=data['name'])
        vocab.token2idx = data['token2idx']
        vocab.idx2token = data['idx2token']
        vocab.token_counts = Counter(data['token_counts'])
        
        print(f"Loaded vocabulary from {path} with {len(vocab)} tokens")
        return vocab


def tokenize(text: str) -> List[str]:
    """
    Improved tokenizer for docstrings.
    Lowercases and splits on whitespace + punctuation boundaries.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    import re
    text = text.strip().lower()
    # Split on whitespace and keep punctuation as separate tokens
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[^\s]", text)
    return tokens


def tokenize_code(code: str) -> List[str]:
    """
    Improved Python code tokenizer.
    Preserves code structure and splits operators/punctuation into separate tokens.
    
    Args:
        code: Python code string
        
    Returns:
        List of code tokens
    """
    import re
    
    # Normalize whitespace: replace newlines and tabs with special tokens
    code = code.replace('\n', ' <NEWLINE> ')
    code = code.replace('\t', ' <INDENT> ')
    code = code.replace('    ', ' <INDENT> ')  # 4-space indent
    
    # Split into meaningful tokens: identifiers, numbers, strings, operators
    tokens = re.findall(
        r'<NEWLINE>|<INDENT>'           # special tokens
        r'|\"\"\".*?\"\"\"|\'\'\'.*?\'\'\''  # triple-quoted strings
        r'|"[^"]*"|\'[^\']*\''          # single/double-quoted strings
        r'|[a-zA-Z_][a-zA-Z0-9_]*'     # identifiers
        r'|[0-9]+\.?[0-9]*'            # numbers
        r'|==|!=|<=|>=|<<|>>|\*\*'     # multi-char operators
        r'|\+=|-=|\*=|/=|//=|%='       # augmented assignment
        r'|[^\s]',                      # any other single char
        code
    )
    
    # Filter empty tokens
    tokens = [t for t in tokens if t.strip()]
    
    return tokens
