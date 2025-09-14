import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
import logging

class WordEmbedding(nn.Module):
    """Word embedding layer with vocabulary mapping."""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Simple word-to-index mapping (in practice, would use pre-trained GloVe)
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def build_vocab(self, tokens_list: List[List[str]]):
        """Build vocabulary from token lists."""
        vocab = set()
        for tokens in tokens_list:
            vocab.update(tokens)
        
        # Add special tokens
        vocab.update(["[PAD]", "[UNK]"])
        
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Update embedding layer if vocab size changed
        if len(self.word_to_idx) != self.vocab_size:
            self.vocab_size = len(self.word_to_idx)
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
    
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices."""
        unk_idx = self.word_to_idx.get("[UNK]", 0)
        return [self.word_to_idx.get(token, unk_idx) for token in tokens]
    
    def forward(self, tokens: List[List[str]]) -> torch.Tensor:
        """
        Forward pass for word embeddings.
        
        Args:
            tokens: List of token sequences [B, L]
            
        Returns:
            Word embeddings [B, L, D_emb]
        """
        # Convert tokens to indices
        batch_indices = []
        for token_seq in tokens:
            indices = self.tokens_to_ids(token_seq)
            batch_indices.append(indices)
        
        # Convert to tensor
        input_ids = torch.tensor(batch_indices, dtype=torch.long, device=self.embedding.weight.device)
        
        return self.embedding(input_ids)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-learnable)."""
    
    def __init__(self, max_length: int, d_model: int):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encodings for sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encodings [seq_len, d_model]
        """
        return self.pe[:seq_len]


class ReasonEmbedding(nn.Module):
    """Reason embedding for em/noem labels."""
    
    def __init__(self, num_labels: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_labels, d_model)
    
    def forward(self, label_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for reason embeddings.
        
        Args:
            label_ids: Label IDs [B, L]
            
        Returns:
            Reason embeddings [B, L, D]
        """
        return self.embedding(label_ids)
